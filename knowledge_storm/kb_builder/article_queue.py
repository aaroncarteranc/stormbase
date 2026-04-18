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
