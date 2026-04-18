# KB Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `knowledge_storm/kb_builder/` — an autonomous overnight orchestrator that wraps Co-STORM to produce a three-tier Obsidian knowledge base (original documents, indexed articles, discussion themes) from a single broad topic.

**Architecture:** A thin `KBOrchestrator` layer above the unmodified `CoStormRunner`. It manages an `ArticleQueue`, runs one Co-STORM session per article with hybrid completion detection, expands the queue via gap analysis after each article, and exports results to an Obsidian vault with frontmatter and wikilinks.

**Tech Stack:** Python 3.11, dspy, trafilatura (fetch + HTML→Markdown, already in requirements.txt), pytest + unittest.mock for tests.

---

## File Map

**Create:**
- `knowledge_storm/kb_builder/__init__.py` — public exports
- `knowledge_storm/kb_builder/article_queue.py` — `ArticleEntry` dataclass + `ArticleQueue` with disk persistence
- `knowledge_storm/kb_builder/page_fetcher.py` — URL fetch + HTML→Markdown via trafilatura, URL-level cache
- `knowledge_storm/kb_builder/completion_checker.py` — hybrid per-article stopping logic (floor + interval + ceiling)
- `knowledge_storm/kb_builder/expansion_planner.py` — post-article gap analysis, curiosity flagging, theme identification, KB sufficiency check
- `knowledge_storm/kb_builder/obsidian_exporter.py` — writes vault files with frontmatter/wikilinks, manages `kb-index.md`
- `knowledge_storm/kb_builder/orchestrator.py` — `KBBuilderConfig` + `KBOrchestrator` main run loop
- `examples/kb_builder_examples/run_kb_builder.py` — end-to-end example script
- `tests/kb_builder/__init__.py`
- `tests/kb_builder/test_article_queue.py`
- `tests/kb_builder/test_page_fetcher.py`
- `tests/kb_builder/test_completion_checker.py`
- `tests/kb_builder/test_expansion_planner.py`
- `tests/kb_builder/test_obsidian_exporter.py`
- `tests/kb_builder/test_orchestrator.py`

---

## Task 1: Package Scaffold

**Files:**
- Create: `knowledge_storm/kb_builder/__init__.py`
- Create: `tests/kb_builder/__init__.py`
- Create: `tests/kb_builder/test_article_queue.py` (stub)

- [ ] **Step 1: Create the kb_builder package**

```bash
mkdir -p knowledge_storm/kb_builder
touch knowledge_storm/kb_builder/__init__.py
mkdir -p tests/kb_builder
touch tests/kb_builder/__init__.py
mkdir -p examples/kb_builder_examples
```

- [ ] **Step 2: Add pytest to requirements if missing**

Check `requirements.txt` — add `pytest` on a new line if not present.

- [ ] **Step 3: Write a canary test to verify pytest works**

`tests/kb_builder/test_article_queue.py`:
```python
def test_canary():
    assert True
```

- [ ] **Step 4: Run canary test**

```bash
pytest tests/kb_builder/test_article_queue.py -v
```
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add knowledge_storm/kb_builder/__init__.py tests/kb_builder/__init__.py tests/kb_builder/test_article_queue.py examples/kb_builder_examples/ requirements.txt
git commit -m "feat: scaffold kb_builder package and test structure"
```

---

## Task 2: ArticleEntry + ArticleQueue

**Files:**
- Create: `knowledge_storm/kb_builder/article_queue.py`
- Modify: `tests/kb_builder/test_article_queue.py`

- [ ] **Step 1: Write the failing tests**

Replace `tests/kb_builder/test_article_queue.py` with:

```python
import json
import os
import pytest
from knowledge_storm.kb_builder.article_queue import ArticleEntry, ArticleQueue


def make_entry(title="A", priority="medium", status="pending"):
    return ArticleEntry(title=title, description="desc", source="seed", priority=priority, status=status)


def test_push_and_pop(tmp_path):
    q = ArticleQueue(str(tmp_path / "queue.json"))
    q.push([make_entry("Alpha")])
    entry = q.pop()
    assert entry.title == "Alpha"
    assert entry.status == "in_progress"


def test_priority_ordering(tmp_path):
    q = ArticleQueue(str(tmp_path / "queue.json"))
    q.push([make_entry("Low", "low"), make_entry("High", "high"), make_entry("Med", "medium")])
    assert q.pop().title == "High"
    assert q.pop().title == "Med"
    assert q.pop().title == "Low"


def test_deduplication_on_push(tmp_path):
    q = ArticleQueue(str(tmp_path / "queue.json"))
    q.push([make_entry("Alpha")])
    q.push([make_entry("Alpha")])  # duplicate
    q.pop()  # in_progress
    assert q.pop() is None  # no second entry


def test_complete_marks_entry(tmp_path):
    q = ArticleQueue(str(tmp_path / "queue.json"))
    q.push([make_entry("Alpha")])
    q.pop()
    q.complete("Alpha")
    assert q.get_completed_titles() == ["Alpha"]


def test_is_empty_when_all_done(tmp_path):
    q = ArticleQueue(str(tmp_path / "queue.json"))
    q.push([make_entry("Alpha")])
    q.pop()
    q.complete("Alpha")
    assert q.is_empty()


def test_persistence(tmp_path):
    path = str(tmp_path / "queue.json")
    q = ArticleQueue(path)
    q.push([make_entry("Alpha")])
    # Reload from disk
    q2 = ArticleQueue(path)
    entry = q2.pop()
    assert entry.title == "Alpha"


def test_normalize_deduplication(tmp_path):
    q = ArticleQueue(str(tmp_path / "queue.json"))
    q.push([make_entry("The JCPOA Agreement")])
    q.push([make_entry("jcpoa agreement")])  # same normalized
    assert q.total_count() == 1


def test_complete_raises_on_missing(tmp_path):
    q = ArticleQueue(str(tmp_path / "queue.json"))
    with pytest.raises(KeyError):
        q.complete("Nonexistent")
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/kb_builder/test_article_queue.py -v
```
Expected: `ImportError` — module does not exist yet.

- [ ] **Step 3: Implement `article_queue.py`**

`knowledge_storm/kb_builder/article_queue.py`:
```python
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import List, Optional

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
STOP_WORDS = {"the", "a", "an", "of", "in", "on", "and", "or", "for", "to", "with"}


def _normalize_title(title: str) -> str:
    words = re.sub(r"[^\w\s]", "", title.lower()).split()
    return " ".join(w for w in words if w not in STOP_WORDS)


@dataclass
class ArticleEntry:
    title: str
    description: str
    source: str   # "seed", "gap_analysis", "curiosity"
    priority: str  # "high", "medium", "low"
    status: str = "pending"  # "pending", "in_progress", "completed"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ArticleEntry":
        return cls(**data)


class ArticleQueue:
    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        self._entries: List[ArticleEntry] = []
        if os.path.exists(persist_path):
            self._load()

    def push(self, entries: List[ArticleEntry]):
        existing_normalized = {_normalize_title(e.title) for e in self._entries}
        for entry in entries:
            if _normalize_title(entry.title) not in existing_normalized:
                self._entries.append(entry)
                existing_normalized.add(_normalize_title(entry.title))
        self._sort()
        self._save()

    def pop(self) -> Optional[ArticleEntry]:
        for entry in self._entries:
            if entry.status == "pending":
                entry.status = "in_progress"
                self._save()
                return entry
        return None

    def complete(self, title: str):
        for entry in self._entries:
            if entry.title == title:
                entry.status = "completed"
                self._save()
                return
        raise KeyError(f"Article not found in queue: {title!r}")

    def is_empty(self) -> bool:
        return all(e.status != "pending" for e in self._entries)

    def total_count(self) -> int:
        return len(self._entries)

    def completed_count(self) -> int:
        return sum(1 for e in self._entries if e.status == "completed")

    def get_all_titles(self) -> List[str]:
        return [e.title for e in self._entries]

    def get_completed_titles(self) -> List[str]:
        return [e.title for e in self._entries if e.status == "completed"]

    def _sort(self):
        self._entries.sort(key=lambda e: PRIORITY_ORDER.get(e.priority, 1))

    def _save(self):
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump([e.to_dict() for e in self._entries], f, indent=2)

    def _load(self):
        with open(self.persist_path) as f:
            data = json.load(f)
        self._entries = [ArticleEntry.from_dict(d) for d in data]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/kb_builder/test_article_queue.py -v
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add knowledge_storm/kb_builder/article_queue.py tests/kb_builder/test_article_queue.py
git commit -m "feat: add ArticleEntry and ArticleQueue with priority ordering and persistence"
```

---

## Task 3: PageFetcher

**Files:**
- Create: `knowledge_storm/kb_builder/page_fetcher.py`
- Create: `tests/kb_builder/test_page_fetcher.py`

- [ ] **Step 1: Write failing tests**

`tests/kb_builder/test_page_fetcher.py`:
```python
import os
from unittest.mock import patch, MagicMock
from knowledge_storm.kb_builder.page_fetcher import PageFetcher


def test_fetch_returns_content_on_success(tmp_path):
    fetcher = PageFetcher(str(tmp_path / "cache"))
    with patch("trafilatura.fetch_url", return_value="<html><body><p>Hello world</p></body></html>"):
        with patch("trafilatura.extract", return_value="Hello world"):
            result = fetcher.fetch("https://example.com/page", fallback_snippet="fallback")
    assert result == "Hello world"


def test_fetch_returns_fallback_on_network_failure(tmp_path):
    fetcher = PageFetcher(str(tmp_path / "cache"))
    with patch("trafilatura.fetch_url", return_value=None):
        result = fetcher.fetch("https://example.com/fail", fallback_snippet="my snippet")
    assert result == "my snippet"


def test_fetch_returns_fallback_on_exception(tmp_path):
    fetcher = PageFetcher(str(tmp_path / "cache"))
    with patch("trafilatura.fetch_url", side_effect=Exception("timeout")):
        result = fetcher.fetch("https://example.com/err", fallback_snippet="fallback text")
    assert result == "fallback text"


def test_second_fetch_uses_cache(tmp_path):
    fetcher = PageFetcher(str(tmp_path / "cache"))
    with patch("trafilatura.fetch_url", return_value="<html>content</html>") as mock_fetch:
        with patch("trafilatura.extract", return_value="content"):
            fetcher.fetch("https://example.com/page", fallback_snippet="")
            fetcher.fetch("https://example.com/page", fallback_snippet="")
    assert mock_fetch.call_count == 1  # fetched once, served from cache second time


def test_already_fetched(tmp_path):
    fetcher = PageFetcher(str(tmp_path / "cache"))
    assert not fetcher.already_fetched("https://example.com/page")
    with patch("trafilatura.fetch_url", return_value="<html></html>"):
        with patch("trafilatura.extract", return_value="text"):
            fetcher.fetch("https://example.com/page", fallback_snippet="")
    assert fetcher.already_fetched("https://example.com/page")


def test_slugify_produces_safe_filename(tmp_path):
    fetcher = PageFetcher(str(tmp_path / "cache"))
    slug = fetcher._slugify("https://example.com/article?id=123&ref=foo")
    assert "/" not in slug
    assert "?" not in slug
    assert "&" not in slug
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/kb_builder/test_page_fetcher.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `page_fetcher.py`**

`knowledge_storm/kb_builder/page_fetcher.py`:
```python
import os
import re
from typing import Dict

import trafilatura


class PageFetcher:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self._url_to_path: Dict[str, str] = {}
        os.makedirs(cache_dir, exist_ok=True)

    def fetch(self, url: str, fallback_snippet: str = "") -> str:
        if url in self._url_to_path:
            with open(self._url_to_path[url], encoding="utf-8") as f:
                return f.read()

        content = self._fetch_url(url)
        if content is None:
            return fallback_snippet

        cache_path = os.path.join(self.cache_dir, self._slugify(url) + ".md")
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)
        self._url_to_path[url] = cache_path
        return content

    def already_fetched(self, url: str) -> bool:
        return url in self._url_to_path

    def _fetch_url(self, url: str):
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                return None
            return trafilatura.extract(
                downloaded,
                output_format="markdown",
                include_links=False,
                include_images=False,
                include_formatting=True,
            )
        except Exception:
            return None

    def _slugify(self, text: str) -> str:
        text = re.sub(r"https?://", "", text)
        text = re.sub(r"[^\w\-]", "-", text)
        return text[:80].strip("-")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/kb_builder/test_page_fetcher.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add knowledge_storm/kb_builder/page_fetcher.py tests/kb_builder/test_page_fetcher.py
git commit -m "feat: add PageFetcher with trafilatura HTML-to-markdown and URL cache"
```

---

## Task 4: CompletionChecker

**Files:**
- Create: `knowledge_storm/kb_builder/completion_checker.py`
- Create: `tests/kb_builder/test_completion_checker.py`

- [ ] **Step 1: Write failing tests**

`tests/kb_builder/test_completion_checker.py`:
```python
from unittest.mock import MagicMock, patch
from knowledge_storm.kb_builder.completion_checker import CompletionChecker


def make_checker(min_floor=10, check_interval=5, max_ceiling=40):
    lm = MagicMock()
    return CompletionChecker(lm=lm, min_floor=min_floor, check_interval=check_interval, max_ceiling=max_ceiling)


def test_should_check_below_floor():
    checker = make_checker(min_floor=10)
    assert not checker.should_check(0)
    assert not checker.should_check(9)


def test_should_check_at_floor():
    checker = make_checker(min_floor=10, check_interval=5)
    assert checker.should_check(10)  # first check at floor


def test_should_check_at_intervals():
    checker = make_checker(min_floor=10, check_interval=5)
    assert checker.should_check(15)
    assert checker.should_check(20)
    assert not checker.should_check(11)
    assert not checker.should_check(14)


def test_is_sufficient_yes(tmp_path):
    lm = MagicMock()
    checker = CompletionChecker(lm=lm, min_floor=10, check_interval=5, max_ceiling=40)

    mock_kb = MagicMock()
    mock_kb.get_node_hierarchy_string.return_value = "# Section A\n## Sub 1\n## Sub 2"

    mock_result = MagicMock()
    mock_result.answer = "yes, the structure looks complete"

    with patch("dspy.Predict.__call__", return_value=mock_result):
        with patch("dspy.settings.context"):
            # Bypass context manager
            checker._checker = MagicMock(return_value=mock_result)
            result = checker.is_sufficient(mock_kb, "JCPOA Agreement")

    assert result is True


def test_is_sufficient_no(tmp_path):
    lm = MagicMock()
    checker = CompletionChecker(lm=lm, min_floor=10, check_interval=5, max_ceiling=40)

    mock_kb = MagicMock()
    mock_kb.get_node_hierarchy_string.return_value = "# Section A"

    mock_result = MagicMock()
    mock_result.answer = "no, coverage is shallow"

    checker._checker = MagicMock(return_value=mock_result)

    with patch("dspy.settings.context"):
        result = checker.is_sufficient(mock_kb, "JCPOA Agreement")

    assert result is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/kb_builder/test_completion_checker.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `completion_checker.py`**

`knowledge_storm/kb_builder/completion_checker.py`:
```python
import dspy
from ..dataclass import KnowledgeBase


class ArticleSufficiencyCheck(dspy.Signature):
    """Assess whether a knowledge base has sufficient coverage to write a complete, well-sourced article.
    Answer 'yes' only if the structure shows broad coverage across multiple distinct subtopics.
    Answer 'no' if important aspects appear missing or coverage is shallow.
    """

    topic = dspy.InputField(prefix="Article topic: ", format=str)
    kb_structure = dspy.InputField(prefix="Knowledge base structure:\n", format=str)
    answer = dspy.OutputField(
        prefix="Sufficient coverage? Answer 'yes' or 'no':\n", format=str
    )


class CompletionChecker:
    def __init__(
        self,
        lm,
        min_floor: int = 10,
        check_interval: int = 5,
        max_ceiling: int = 40,
    ):
        self.lm = lm
        self.min_floor = min_floor
        self.check_interval = check_interval
        self.max_ceiling = max_ceiling
        self._checker = dspy.Predict(ArticleSufficiencyCheck)

    def should_check(self, turn_number: int) -> bool:
        if turn_number < self.min_floor:
            return False
        return (turn_number - self.min_floor) % self.check_interval == 0

    def is_sufficient(self, knowledge_base: KnowledgeBase, article_title: str) -> bool:
        structure = knowledge_base.get_node_hierarchy_string(
            include_indent=False,
            include_full_path=False,
            include_hash_tag=True,
            include_node_content_count=False,
        )
        with dspy.settings.context(lm=self.lm, show_guidelines=False):
            result = self._checker(topic=article_title, kb_structure=structure)
        return result.answer.strip().lower().startswith("yes")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/kb_builder/test_completion_checker.py -v
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add knowledge_storm/kb_builder/completion_checker.py tests/kb_builder/test_completion_checker.py
git commit -m "feat: add CompletionChecker with hybrid floor/interval/ceiling stopping logic"
```

---

## Task 5: ExpansionPlanner

**Files:**
- Create: `knowledge_storm/kb_builder/expansion_planner.py`
- Create: `tests/kb_builder/test_expansion_planner.py`

- [ ] **Step 1: Write failing tests**

`tests/kb_builder/test_expansion_planner.py`:
```python
from unittest.mock import MagicMock, patch
from knowledge_storm.kb_builder.expansion_planner import ExpansionPlanner
from knowledge_storm.kb_builder.article_queue import ArticleEntry


def make_planner():
    lm = MagicMock()
    return ExpansionPlanner(lm=lm, broad_topic="US-Iran Relations")


def test_parse_proposals_valid_json():
    planner = make_planner()
    raw = '[{"title": "JCPOA History", "description": "Background", "priority": "high"}]'
    entries = planner._parse_proposals(raw)
    assert len(entries) == 1
    assert entries[0].title == "JCPOA History"
    assert entries[0].priority == "high"
    assert entries[0].source == "gap_analysis"


def test_parse_proposals_caps_at_three():
    planner = make_planner()
    raw = '[{"title":"A","description":"","priority":"high"},{"title":"B","description":"","priority":"medium"},{"title":"C","description":"","priority":"low"},{"title":"D","description":"","priority":"low"}]'
    entries = planner._parse_proposals(raw)
    assert len(entries) == 3


def test_parse_proposals_invalid_json_returns_empty():
    planner = make_planner()
    entries = planner._parse_proposals("I suggest writing about JCPOA.")
    assert entries == []


def test_extract_json_strips_code_fences():
    planner = make_planner()
    raw = '```json\n[{"title": "Test"}]\n```'
    result = planner._extract_json(raw)
    assert result.strip() == '[{"title": "Test"}]'


def test_flag_curiosity_candidate_deduplication():
    planner = make_planner()
    planner.flag_curiosity_candidate("Ayatollah Khamenei")
    planner.flag_curiosity_candidate("Ayatollah Khamenei")
    assert planner._curiosity_candidates.count("Ayatollah Khamenei") == 1


def test_flush_curiosity_candidates_clears_list():
    planner = make_planner()
    planner.flag_curiosity_candidate("Nuclear Deal")
    candidates = planner.flush_curiosity_candidates()
    assert "Nuclear Deal" in candidates
    assert planner._curiosity_candidates == []


def test_kb_is_sufficient_yes():
    planner = make_planner()
    mock_result = MagicMock()
    mock_result.answer = "yes, comprehensive"
    planner._kb_check = MagicMock(return_value=mock_result)
    with patch("dspy.settings.context"):
        result = planner.kb_is_sufficient("- [[indexed/jcpoa]] — Nuclear deal\n")
    assert result is True


def test_kb_is_sufficient_no():
    planner = make_planner()
    mock_result = MagicMock()
    mock_result.answer = "no, missing sanctions history"
    planner._kb_check = MagicMock(return_value=mock_result)
    with patch("dspy.settings.context"):
        result = planner.kb_is_sufficient("- [[indexed/jcpoa]] — Nuclear deal\n")
    assert result is False


def test_parse_themes_valid():
    planner = make_planner()
    raw = '[{"theme": "Sanctions Diplomacy", "description": "How sanctions shaped talks", "relevant_articles": ["JCPOA", "2018 Withdrawal"]}]'
    themes = planner._parse_themes(raw)
    assert len(themes) == 1
    assert themes[0]["theme"] == "Sanctions Diplomacy"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/kb_builder/test_expansion_planner.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `expansion_planner.py`**

`knowledge_storm/kb_builder/expansion_planner.py`:
```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/kb_builder/test_expansion_planner.py -v
```
Expected: `10 passed`

- [ ] **Step 5: Commit**

```bash
git add knowledge_storm/kb_builder/expansion_planner.py tests/kb_builder/test_expansion_planner.py
git commit -m "feat: add ExpansionPlanner with gap analysis, curiosity flagging, and theme identification"
```

---

## Task 6: ObsidianExporter

**Files:**
- Create: `knowledge_storm/kb_builder/obsidian_exporter.py`
- Create: `tests/kb_builder/test_obsidian_exporter.py`

- [ ] **Step 1: Write failing tests**

`tests/kb_builder/test_obsidian_exporter.py`:
```python
import os
from unittest.mock import MagicMock, patch
from knowledge_storm.kb_builder.obsidian_exporter import ObsidianExporter
from knowledge_storm.kb_builder.page_fetcher import PageFetcher
from knowledge_storm.interface import Information


def make_exporter(tmp_path):
    fetcher = MagicMock(spec=PageFetcher)
    fetcher.fetch.return_value = "Full page content here."
    lm = MagicMock()
    return ObsidianExporter(vault_dir=str(tmp_path), page_fetcher=fetcher, lm=lm)


def make_info(url="https://example.com/article", title="Example Article", citation_uuid=1):
    info = Information(url=url, description="desc", snippets=["snippet"], title=title)
    info.citation_uuid = citation_uuid
    return info


def test_export_indexed_doc_creates_file(tmp_path):
    exporter = make_exporter(tmp_path)
    report = "The JCPOA was signed in 2015 [1]. It involved six nations [2]."
    infos = [make_info(citation_uuid=1), make_info(url="https://other.com", title="Other", citation_uuid=2)]
    path = exporter.export_indexed_doc("JCPOA Overview", "US-Iran Relations", report, infos)
    assert os.path.exists(path)
    content = open(path).read()
    assert "title: JCPOA Overview" in content
    assert "topic: US-Iran Relations" in content
    assert "[[original-documents/" in content


def test_citation_replacement(tmp_path):
    exporter = make_exporter(tmp_path)
    citation_map = {1: "example-article", 2: "other-source"}
    result = exporter._replace_citations_with_wikilinks("Text [1] and [2] here.", citation_map)
    assert "[[original-documents/example-article]]" in result
    assert "[[original-documents/other-source]]" in result
    assert "[1]" not in result


def test_unknown_citation_preserved(tmp_path):
    exporter = make_exporter(tmp_path)
    citation_map = {1: "source-a"}
    result = exporter._replace_citations_with_wikilinks("Text [1] and [99] here.", citation_map)
    assert "[99]" in result  # unknown citation is left as-is


def test_export_original_doc_creates_file(tmp_path):
    exporter = make_exporter(tmp_path)
    info = make_info()
    path = exporter.export_original_doc(info)
    assert os.path.exists(path)
    content = open(path).read()
    assert "url: https://example.com/article" in content
    assert "Full page content here." in content


def test_export_original_doc_deduplicates(tmp_path):
    exporter = make_exporter(tmp_path)
    info = make_info()
    path1 = exporter.export_original_doc(info)
    path2 = exporter.export_original_doc(info)
    assert path1 == path2
    assert exporter.page_fetcher.fetch.call_count == 1


def test_update_kb_index_appends(tmp_path):
    exporter = make_exporter(tmp_path)
    exporter.update_kb_index("JCPOA Overview", "The 2015 nuclear agreement between Iran and world powers.")
    index = exporter.read_kb_index()
    assert "[[indexed/jcpoa-overview]]" in index
    assert "The 2015 nuclear agreement" in index


def test_read_kb_index_empty(tmp_path):
    exporter = make_exporter(tmp_path)
    assert exporter.read_kb_index() == ""


def test_slugify(tmp_path):
    exporter = make_exporter(tmp_path)
    assert exporter._slugify("The JCPOA Agreement!") == "the-jcpoa-agreement"
    assert exporter._slugify("US–Iran Relations (2003–2024)") == "us-iran-relations-2003-2024"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/kb_builder/test_obsidian_exporter.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `obsidian_exporter.py`**

`knowledge_storm/kb_builder/obsidian_exporter.py`:
```python
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
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_]+", "-", text)
        return text[:80].strip("-")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/kb_builder/test_obsidian_exporter.py -v
```
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add knowledge_storm/kb_builder/obsidian_exporter.py tests/kb_builder/test_obsidian_exporter.py
git commit -m "feat: add ObsidianExporter with frontmatter, wikilink citation replacement, and kb-index"
```

---

## Task 7: KBOrchestrator

**Files:**
- Create: `knowledge_storm/kb_builder/orchestrator.py`
- Create: `tests/kb_builder/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

`tests/kb_builder/test_orchestrator.py`:
```python
from unittest.mock import MagicMock, patch
from knowledge_storm.kb_builder.orchestrator import KBOrchestrator, KBBuilderConfig
from knowledge_storm.kb_builder.article_queue import ArticleEntry


def make_config(tmp_path):
    return KBBuilderConfig(
        topic="US-Iran Relations",
        output_dir=str(tmp_path),
        max_articles=10,
        max_expansion_rounds=3,
        min_floor=2,
        check_interval=2,
        max_ceiling=5,
        completion_silence_n=2,
    )


def make_orchestrator(tmp_path):
    config = make_config(tmp_path)
    lm_config = MagicMock()
    lm_config.knowledge_base_lm = MagicMock()
    rm = MagicMock()
    return KBOrchestrator(config=config, lm_config=lm_config, rm=rm)


def test_scan_for_curiosity_candidates_finds_proper_nouns():
    orch = make_orchestrator(__import__("tempfile").mkdtemp())
    candidates = orch._scan_for_curiosity_candidates(
        "The Islamic Republic and Supreme Leader Khamenei discussed Nuclear Program details.",
        known_titles=set()
    )
    assert any("Khamenei" in c or "Nuclear Program" in c or "Islamic Republic" in c for c in candidates)


def test_scan_for_curiosity_candidates_skips_known(tmp_path):
    orch = make_orchestrator(tmp_path)
    candidates = orch._scan_for_curiosity_candidates(
        "Nuclear Program was the focus.",
        known_titles={"Nuclear Program"}
    )
    assert "Nuclear Program" not in candidates


def test_kb_is_complete_expansion_ceiling(tmp_path):
    orch = make_orchestrator(tmp_path)
    # Hits max_expansion_rounds
    result = orch._kb_is_complete(expansion_round=3)
    assert result is True


def test_kb_is_complete_article_ceiling(tmp_path):
    orch = make_orchestrator(tmp_path)
    # Queue has max_articles entries
    orch.queue.push([
        ArticleEntry(title=f"Article {i}", description="", source="seed", priority="medium")
        for i in range(10)
    ])
    result = orch._kb_is_complete(expansion_round=0)
    assert result is True


def test_kb_is_complete_silence_triggers_semantic_check(tmp_path):
    orch = make_orchestrator(tmp_path)
    orch._silence_count = 2  # equals completion_silence_n
    orch.expansion_planner.kb_is_sufficient = MagicMock(return_value=True)
    result = orch._kb_is_complete(expansion_round=0)
    assert result is True
    orch.expansion_planner.kb_is_sufficient.assert_called_once()


def test_kb_is_complete_silence_but_not_sufficient(tmp_path):
    orch = make_orchestrator(tmp_path)
    orch._silence_count = 2
    orch.expansion_planner.kb_is_sufficient = MagicMock(return_value=False)
    result = orch._kb_is_complete(expansion_round=0)
    assert result is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/kb_builder/test_orchestrator.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement `orchestrator.py`**

`knowledge_storm/kb_builder/orchestrator.py`:
```python
import os
import re
from dataclasses import dataclass, field
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

        expansion_round = 0
        while not self.queue.is_empty():
            if self._kb_is_complete(expansion_round):
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

            summary = self._summarize_article(article.title, report[:500])
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

            expansion_round += 1

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

    def _kb_is_complete(self, expansion_round: int) -> bool:
        if expansion_round >= self.config.max_expansion_rounds:
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/kb_builder/test_orchestrator.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/kb_builder/ -v
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add knowledge_storm/kb_builder/orchestrator.py tests/kb_builder/test_orchestrator.py
git commit -m "feat: add KBOrchestrator with main run loop, expansion, and KB completion detection"
```

---

## Task 8: Wire Up __init__.py and Example Script

**Files:**
- Modify: `knowledge_storm/kb_builder/__init__.py`
- Create: `examples/kb_builder_examples/run_kb_builder.py`

- [ ] **Step 1: Populate `__init__.py`**

`knowledge_storm/kb_builder/__init__.py`:
```python
from .orchestrator import KBOrchestrator, KBBuilderConfig
from .article_queue import ArticleEntry, ArticleQueue

__all__ = ["KBOrchestrator", "KBBuilderConfig", "ArticleEntry", "ArticleQueue"]
```

- [ ] **Step 2: Write the example script**

`examples/kb_builder_examples/run_kb_builder.py`:
```python
"""
Example: Autonomous KB Builder using Ollama + DuckDuckGo

Usage:
    python examples/kb_builder_examples/run_kb_builder.py \
        --topic "US-Iran Relations" \
        --output-dir ./kb_output \
        --model ollama/llama3:70b

Requires: secrets.toml with BING_SEARCH_API_KEY, or use --retriever duckduckgo
"""

import argparse
import os

from knowledge_storm.collaborative_storm.engine import CollaborativeStormLMConfigs
from knowledge_storm.kb_builder import KBBuilderConfig, KBOrchestrator
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import DuckDuckGoSearchRM


def main():
    parser = argparse.ArgumentParser(description="KB Builder — autonomous Obsidian KB generation")
    parser.add_argument("--topic", required=True, help="Broad topic for the knowledge base")
    parser.add_argument("--output-dir", required=True, help="Output directory (Obsidian vault)")
    parser.add_argument(
        "--model",
        default="ollama/llama3:70b",
        help="LiteLLM model string (e.g. ollama/llama3:70b, together_ai/meta-llama/...)",
    )
    parser.add_argument("--max-articles", type=int, default=50)
    parser.add_argument("--max-ceiling", type=int, default=40, help="Max turns per article")
    parser.add_argument("--min-floor", type=int, default=10, help="Min turns before completion checks")
    args = parser.parse_args()

    # LM config — all roles use the same model for local inference
    lm_kwargs = {"model": args.model, "max_tokens": 2000, "model_type": "chat"}
    lm = LitellmModel(**lm_kwargs)

    lm_config = CollaborativeStormLMConfigs()
    lm_config.set_question_answering_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 1000}))
    lm_config.set_discourse_manage_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 500}))
    lm_config.set_utterance_polishing_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 2000}))
    lm_config.set_warmstart_outline_gen_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 500}))
    lm_config.set_question_asking_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 300}))
    lm_config.set_knowledge_base_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 1000}))

    rm = DuckDuckGoSearchRM(k=10)

    config = KBBuilderConfig(
        topic=args.topic,
        output_dir=args.output_dir,
        max_articles=args.max_articles,
        max_ceiling=args.max_ceiling,
        min_floor=args.min_floor,
    )

    print(f"Starting KB Builder for topic: {args.topic!r}")
    print(f"Output vault: {args.output_dir}")
    print(f"Model: {args.model}")

    orchestrator = KBOrchestrator(config=config, lm_config=lm_config, rm=rm)
    orchestrator.run()

    print(f"\nDone. Open {args.output_dir} in Obsidian.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the full test suite one more time**

```bash
pytest tests/kb_builder/ -v
```
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add knowledge_storm/kb_builder/__init__.py examples/kb_builder_examples/run_kb_builder.py
git commit -m "feat: wire up kb_builder public API and add run_kb_builder example script"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Original documents (full HTML→MD via trafilatura) → `PageFetcher` + `ObsidianExporter.export_original_doc`
- [x] Indexed documents with wikilinks + frontmatter → `ObsidianExporter.export_indexed_doc`
- [x] Discussion documents (cross-cutting themes, post-KB) → `ObsidianExporter.export_discussion_doc` + `ExpansionPlanner.identify_discussion_themes`
- [x] Hybrid completion detection (floor + interval + ceiling) → `CompletionChecker`
- [x] Curious LLM expansion (gap analysis + inline flagging) → `ExpansionPlanner.run` + `KBOrchestrator._scan_for_curiosity_candidates`
- [x] KB completion detection (silence + semantic check) → `KBOrchestrator._kb_is_complete`
- [x] Hard ceilings (`max_articles`, `max_expansion_rounds`) → `KBBuilderConfig` + `_kb_is_complete`
- [x] KB index as LLM's window (never full articles) → `kb-index.md` + `ObsidianExporter.read_kb_index`
- [x] Crash recovery via persisted queue → `ArticleQueue` JSON persistence
- [x] URL deduplication for original docs → `ObsidianExporter.export_original_doc` checks `os.path.exists`
- [x] KB index injection as first user utterance → `KBOrchestrator._run_article`
- [x] Title deduplication with normalization → `ArticleQueue._normalize_title`

**Type consistency:** All method names are consistent across tasks. `_parse_proposals` in `ExpansionPlanner` is reused by `KBOrchestrator._seed_articles` — correct.

**No placeholders:** All code steps contain complete, runnable code.
