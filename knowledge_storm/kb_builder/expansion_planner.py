import json
import re
import dspy
from typing import List, Dict, Any

from .article_queue import ArticleEntry


class ProposeArticles(dspy.Signature):
    """Given a broad research topic, a recently completed article, the current knowledge base
    index, and any curiosity candidates surfaced during research, propose 0-3 new articles that
    would most improve KB coverage. Return a JSON array of objects with keys: title, description,
    priority (one of: high, medium, low). Return an empty array [] if no new articles are needed.
    Do not propose articles already listed in the KB index.
    """

    broad_topic = dspy.InputField(prefix="Broad topic: ", format=str)
    finished_article = dspy.InputField(prefix="Finished article title: ", format=str)
    kb_index = dspy.InputField(
        prefix="Current KB index (already covered):\n", format=str
    )
    curiosity_candidates = dspy.InputField(
        prefix="Concepts surfaced during research:\n", format=str
    )
    proposals = dspy.OutputField(
        prefix="Proposed new articles as JSON array:\n", format=str
    )


class KBSufficiencyCheck(dspy.Signature):
    """Assess whether a knowledge base provides thorough, well-rounded coverage of a broad topic.
    Answer 'yes' only if all major aspects are covered. Answer 'no' if significant gaps remain.
    """

    broad_topic = dspy.InputField(prefix="Broad topic: ", format=str)
    kb_index = dspy.InputField(prefix="Knowledge base index:\n", format=str)
    answer = dspy.OutputField(
        prefix="Thorough coverage? Answer 'yes' or 'no' with one sentence rationale:\n",
        format=str,
    )


class IdentifyThemes(dspy.Signature):
    """Given a completed knowledge base index, identify 3-7 cross-cutting themes that span
    multiple articles and would benefit from dedicated synthesis documents. Return a JSON array
    of objects with keys: theme (short title), description (one sentence), relevant_articles
    (list of article titles from the index).
    """

    broad_topic = dspy.InputField(prefix="Broad topic: ", format=str)
    kb_index = dspy.InputField(prefix="Knowledge base index:\n", format=str)
    themes = dspy.OutputField(
        prefix="Cross-cutting themes as JSON array:\n", format=str
    )


class ExpansionPlanner:
    def __init__(self, lm, broad_topic: str):
        self.lm = lm
        self.broad_topic = broad_topic
        self._curiosity_candidates: List[str] = []
        self._propose = dspy.Predict(ProposeArticles)
        self._kb_check = dspy.Predict(KBSufficiencyCheck)
        self._identify_themes = dspy.Predict(IdentifyThemes)

    def flag_curiosity_candidate(self, candidate: str):
        if candidate not in self._curiosity_candidates:
            self._curiosity_candidates.append(candidate)

    def flush_curiosity_candidates(self) -> List[str]:
        candidates = list(self._curiosity_candidates)
        self._curiosity_candidates = []
        return candidates

    def run(
        self,
        finished_article_title: str,
        kb_index: str,
        inline_candidates: List[str],
    ) -> List[ArticleEntry]:
        candidates_str = ", ".join(inline_candidates) if inline_candidates else "none"
        with dspy.settings.context(lm=self.lm, show_guidelines=False):
            result = self._propose(
                broad_topic=self.broad_topic,
                finished_article=finished_article_title,
                kb_index=kb_index,
                curiosity_candidates=candidates_str,
            )
        return self._parse_proposals(result.proposals)

    def kb_is_sufficient(self, kb_index: str) -> bool:
        with dspy.settings.context(lm=self.lm, show_guidelines=False):
            result = self._kb_check(
                broad_topic=self.broad_topic, kb_index=kb_index
            )
        return result.answer.strip().lower().startswith("yes")

    def identify_discussion_themes(self, kb_index: str) -> List[Dict[str, Any]]:
        with dspy.settings.context(lm=self.lm, show_guidelines=False):
            result = self._identify_themes(
                broad_topic=self.broad_topic, kb_index=kb_index
            )
        return self._parse_themes(result.themes)

    def _parse_proposals(self, raw: str) -> List[ArticleEntry]:
        try:
            data = json.loads(self._extract_json(raw))
            entries = []
            for item in data[:3]:
                entries.append(
                    ArticleEntry(
                        title=item["title"],
                        description=item.get("description", ""),
                        source="gap_analysis",
                        priority=item.get("priority", "medium"),
                    )
                )
            return entries
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _parse_themes(self, raw: str) -> List[Dict[str, Any]]:
        try:
            return json.loads(self._extract_json(raw))
        except (json.JSONDecodeError, TypeError):
            return []

    def _extract_json(self, text: str) -> str:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            return match.group(1).strip()
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            return match.group(0)
        return text.strip()
