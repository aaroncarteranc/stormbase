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
