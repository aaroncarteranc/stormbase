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
