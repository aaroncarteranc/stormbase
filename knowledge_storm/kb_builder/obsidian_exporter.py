import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional

import dspy

from ..interface import Information
from .page_fetcher import PageFetcher


class WriteDiscussionDoc(dspy.Signature):
    """Write a synthetic analysis document exploring a cross-cutting theme across multiple
    knowledge base articles. Draw connections, identify patterns, and offer considered
    perspectives. Use [[wikilink]] syntax to reference indexed articles. Write in clear,
    analytical prose. Do not fabricate facts not evidenced by the article titles/descriptions.
    """

    broad_topic = dspy.InputField(prefix="Broad topic: ", format=str)
    theme = dspy.InputField(prefix="Theme: ", format=str)
    theme_description = dspy.InputField(prefix="Theme description: ", format=str)
    relevant_articles = dspy.InputField(prefix="Relevant articles:\n", format=str)
    output = dspy.OutputField(prefix="Discussion document (Markdown):\n", format=str)


class ObsidianExporter:
    def __init__(self, vault_dir: str, page_fetcher: PageFetcher, lm=None):
        self.vault_dir = vault_dir
        self.page_fetcher = page_fetcher
        self.lm = lm
        self._write_discussion = dspy.Predict(WriteDiscussionDoc)
        for subdir in ("original-documents", "indexed", "discussion", "_meta"):
            os.makedirs(os.path.join(vault_dir, subdir), exist_ok=True)

    def export_indexed_doc(
        self,
        article_title: str,
        broad_topic: str,
        report: str,
        info_objects: List[Information],
    ) -> str:
        citation_map = self._build_citation_map(info_objects)
        content = self._replace_citations_with_wikilinks(report, citation_map)
        source_wikilinks = [
            f'  - "[[original-documents/{self._slugify(info.title or info.url)}]]"'
            for info in info_objects
        ]
        frontmatter = (
            "---\n"
            f"title: {article_title}\n"
            f"topic: {broad_topic}\n"
            f"created_at: {datetime.now(timezone.utc).isoformat()}\n"
            "sources:\n"
            + "\n".join(source_wikilinks)
            + "\ntags: []\n---\n\n"
        )
        slug = self._slugify(article_title)
        path = os.path.join(self.vault_dir, "indexed", f"{slug}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(frontmatter + content)
        return path

    def export_original_doc(self, info: Information) -> str:
        slug = self._slugify(info.title or info.url)
        path = os.path.join(self.vault_dir, "original-documents", f"{slug}.md")
        if os.path.exists(path):
            return path
        fallback = info.snippets[0] if info.snippets else ""
        content = self.page_fetcher.fetch(info.url, fallback_snippet=fallback)
        frontmatter = (
            "---\n"
            f"title: {info.title or slug}\n"
            f"url: {info.url}\n"
            f"fetched_at: {datetime.now(timezone.utc).isoformat()}\n"
            "cited_by: []\n"
            "---\n\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(frontmatter + content)
        return path

    def export_discussion_doc(
        self,
        broad_topic: str,
        theme: str,
        theme_description: str,
        relevant_articles: List[str],
    ) -> str:
        articles_str = "\n".join(
            f"- [[indexed/{self._slugify(t)}]] ({t})" for t in relevant_articles
        )
        with dspy.settings.context(lm=self.lm, show_guidelines=False):
            result = self._write_discussion(
                broad_topic=broad_topic,
                theme=theme,
                theme_description=theme_description,
                relevant_articles=articles_str,
            )
        draws_from = "\n".join(
            f'  - "[[indexed/{self._slugify(t)}]]"' for t in relevant_articles
        )
        frontmatter = (
            "---\n"
            f"title: {theme}\n"
            f"theme: {theme_description}\n"
            f"created_at: {datetime.now(timezone.utc).isoformat()}\n"
            "draws_from:\n"
            + draws_from
            + "\n---\n\n"
        )
        slug = self._slugify(theme)
        path = os.path.join(self.vault_dir, "discussion", f"{slug}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(frontmatter + result.output)
        return path

    def update_kb_index(self, article_title: str, one_line_summary: str):
        index_path = os.path.join(self.vault_dir, "_meta", "kb-index.md")
        slug = self._slugify(article_title)
        line = f"- [[indexed/{slug}]] — {one_line_summary}\n"
        with open(index_path, "a", encoding="utf-8") as f:
            f.write(line)

    def read_kb_index(self) -> str:
        index_path = os.path.join(self.vault_dir, "_meta", "kb-index.md")
        if not os.path.exists(index_path):
            return ""
        with open(index_path, encoding="utf-8") as f:
            return f.read()

    def append_run_log(self, message: str):
        log_path = os.path.join(self.vault_dir, "_meta", "run-log.md")
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def _build_citation_map(self, info_objects: List[Information]) -> Dict[int, str]:
        return {
            info.citation_uuid: self._slugify(info.title or info.url)
            for info in info_objects
            if info.citation_uuid >= 0
        }

    def _replace_citations_with_wikilinks(
        self, text: str, citation_map: Dict[int, str]
    ) -> str:
        def replace(match):
            idx = int(match.group(1))
            slug = citation_map.get(idx)
            if slug:
                return f"[[original-documents/{slug}]]"
            return match.group(0)

        return re.sub(r"\[(\d+)\]", replace, text)

    def _slugify(self, text: str) -> str:
        text = text.lower()
        # Replace Unicode dashes with ASCII hyphen so they create word boundaries
        text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", "-", text)
        # Replace non-word, non-space, non-hyphen chars with space
        text = re.sub(r"[^\w\s-]", " ", text)
        # Collapse whitespace and underscores into hyphens
        text = re.sub(r"[\s_]+", "-", text)
        # Collapse multiple hyphens
        text = re.sub(r"-+", "-", text)
        return text[:80].strip("-")
