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
