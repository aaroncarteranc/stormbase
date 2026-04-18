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
