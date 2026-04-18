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
