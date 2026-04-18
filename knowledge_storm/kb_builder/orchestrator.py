import os
import re
from dataclasses import dataclass
from typing import List, Set

import dspy

from ..collaborative_storm.engine import (
    CoStormRunner,
    RunnerArgument,
    CollaborativeStormLMConfigs,
)
from ..logging_wrapper import LoggingWrapper
from .article_queue import ArticleEntry, ArticleQueue
from .completion_checker import CompletionChecker
from .expansion_planner import ExpansionPlanner
from .obsidian_exporter import ObsidianExporter
from .page_fetcher import PageFetcher


class SeedArticles(dspy.Signature):
    """Generate an initial list of 5-10 focused article topics to seed a knowledge base on
    a broad topic. Each article should cover a distinct, well-scoped subtopic. Return a JSON
    array of objects with keys: title, description, priority (high/medium/low).
    """

    broad_topic = dspy.InputField(prefix="Broad topic: ", format=str)
    articles = dspy.OutputField(
        prefix="Initial article list as JSON array:\n", format=str
    )


class ArticleSummary(dspy.Signature):
    """Write a single sentence (max 20 words) summarizing what an article covers,
    suitable for an index entry.
    """

    article_title = dspy.InputField(prefix="Article title: ", format=str)
    report_excerpt = dspy.InputField(prefix="Article opening:\n", format=str)
    summary = dspy.OutputField(prefix="One-line summary:\n", format=str)


@dataclass
class KBBuilderConfig:
    topic: str
    output_dir: str
    max_articles: int = 50
    max_expansion_rounds: int = 10
    min_floor: int = 10
    check_interval: int = 5
    max_ceiling: int = 40
    completion_silence_n: int = 3


class KBOrchestrator:
    def __init__(
        self,
        config: KBBuilderConfig,
        lm_config: CollaborativeStormLMConfigs,
        rm,
    ):
        self.config = config
        self.lm_config = lm_config
        self.rm = rm
        self._silence_count = 0

        self.page_fetcher = PageFetcher(
            cache_dir=os.path.join(config.output_dir, ".page-cache")
        )
        self.exporter = ObsidianExporter(
            vault_dir=config.output_dir,
            page_fetcher=self.page_fetcher,
            lm=lm_config.knowledge_base_lm,
        )
        self.queue = ArticleQueue(
            persist_path=os.path.join(config.output_dir, "_meta", "queue.json")
        )
        self.completion_checker = CompletionChecker(
            lm=lm_config.knowledge_base_lm,
            min_floor=config.min_floor,
            check_interval=config.check_interval,
            max_ceiling=config.max_ceiling,
        )
        self.expansion_planner = ExpansionPlanner(
            lm=lm_config.knowledge_base_lm,
            broad_topic=config.topic,
        )
        self._seed_module = dspy.Predict(SeedArticles)
        self._summary_module = dspy.Predict(ArticleSummary)

    def run(self):
        self.exporter.append_run_log(f"KB Builder started. Topic: {self.config.topic}")

        if self.queue.is_empty() and self.queue.total_count() == 0:
            seed_entries = self._seed_articles()
            self.queue.push(seed_entries)
            self.exporter.append_run_log(
                f"Seeded queue with {len(seed_entries)} articles."
            )

        articles_completed = 0
        while not self.queue.is_empty():
            if self._kb_is_complete(articles_completed):
                self.exporter.append_run_log(
                    "KB completion detected. Proceeding to discussion generation."
                )
                break

            article = self.queue.pop()
            if article is None:
                break

            self.exporter.append_run_log(f"Starting article: {article.title}")
            report, info_objects = self._run_article(article)

            for info in info_objects:
                self.exporter.export_original_doc(info)
            self.exporter.export_indexed_doc(
                article_title=article.title,
                broad_topic=self.config.topic,
                report=report,
                info_objects=info_objects,
            )

            summary = self._summarize_article(article.title, report[:500]) if report else article.description
            self.exporter.update_kb_index(article.title, summary)
            self.queue.complete(article.title)
            self.exporter.append_run_log(f"Completed article: {article.title}")

            if self.queue.total_count() < self.config.max_articles:
                kb_index = self.exporter.read_kb_index()
                inline_candidates = self.expansion_planner.flush_curiosity_candidates()
                new_entries = self.expansion_planner.run(
                    finished_article_title=article.title,
                    kb_index=kb_index,
                    inline_candidates=inline_candidates,
                )
                if new_entries:
                    self._silence_count = 0
                    self.queue.push(new_entries)
                    self.exporter.append_run_log(
                        f"Expansion: added {len(new_entries)} new articles."
                    )
                else:
                    self._silence_count += 1

            articles_completed += 1

        self._generate_discussion_docs()
        self.exporter.append_run_log("KB Builder run complete.")

    def _seed_articles(self) -> List[ArticleEntry]:
        with dspy.settings.context(lm=self.lm_config.knowledge_base_lm, show_guidelines=False):
            result = self._seed_module(broad_topic=self.config.topic)
        return self.expansion_planner._parse_proposals(result.articles)

    def _run_article(self, article: ArticleEntry):
        runner_arg = RunnerArgument(
            topic=f"{self.config.topic} — {article.title}",
            total_conv_turn=self.config.max_ceiling,
        )
        logging_wrapper = LoggingWrapper(self.lm_config)
        runner = CoStormRunner(
            lm_config=self.lm_config,
            runner_argument=runner_arg,
            logging_wrapper=logging_wrapper,
            rm=self.rm,
        )
        runner.warm_start()

        kb_index = self.exporter.read_kb_index()
        if kb_index:
            runner.step(
                user_utterance=(
                    "Here is what the knowledge base already covers — "
                    f"please avoid redundancy with these topics:\n{kb_index}"
                )
            )

        known_titles: Set[str] = set(self.queue.get_all_titles())

        for turn_num in range(self.config.max_ceiling):
            conv_turn = runner.step(simulate_user=True, simulate_user_intent="")
            if conv_turn and conv_turn.utterance:
                candidates = self._scan_for_curiosity_candidates(
                    conv_turn.utterance, known_titles
                )
                for c in candidates:
                    self.expansion_planner.flag_curiosity_candidate(c)

            if self.completion_checker.should_check(turn_num):
                if self.completion_checker.is_sufficient(
                    runner.knowledge_base, article.title
                ):
                    break

        report = runner.generate_report()
        info_objects = list(runner.knowledge_base.info_uuid_to_info_dict.values())
        return report, info_objects

    def _scan_for_curiosity_candidates(
        self, utterance: str, known_titles: Set[str]
    ) -> List[str]:
        phrases = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", utterance)
        normalized_known = {t.lower() for t in known_titles}
        candidates = []
        for phrase in phrases:
            if phrase.lower() not in normalized_known and len(phrase) > 4:
                candidates.append(phrase)
        return candidates[:5]

    def _kb_is_complete(self, articles_completed: int) -> bool:
        if articles_completed >= self.config.max_expansion_rounds:
            return True
        if self.queue.total_count() >= self.config.max_articles:
            return True
        if self._silence_count >= self.config.completion_silence_n:
            kb_index = self.exporter.read_kb_index()
            return self.expansion_planner.kb_is_sufficient(kb_index)
        return False

    def _generate_discussion_docs(self):
        kb_index = self.exporter.read_kb_index()
        themes = self.expansion_planner.identify_discussion_themes(kb_index)
        self.exporter.append_run_log(
            f"Generating {len(themes)} discussion documents."
        )
        for theme_data in themes:
            self.exporter.export_discussion_doc(
                broad_topic=self.config.topic,
                theme=theme_data.get("theme", ""),
                theme_description=theme_data.get("description", ""),
                relevant_articles=theme_data.get("relevant_articles", []),
            )

    def _summarize_article(self, title: str, excerpt: str) -> str:
        with dspy.settings.context(lm=self.lm_config.knowledge_base_lm, show_guidelines=False):
            result = self._summary_module(article_title=title, report_excerpt=excerpt)
        return result.summary.strip()
