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
