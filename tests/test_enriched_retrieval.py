from pathlib import Path

import pytest
from langchain_core.documents import Document

import rag
import tools


def test_context_preserves_table_label_and_page_range() -> None:
    doc = Document(
        page_content="[TABLE]\n|A|B|",
        metadata={
            "source": "guide.pdf",
            "page_start": 4,
            "page_end": 5,
            "content_type": "table",
        },
    )

    context = rag.format_context([doc])

    assert "guide.pdf, pages 4-5" in context
    assert "[TABLE]" in context


def test_source_formatting_migrates_legacy_zero_based_page_metadata() -> None:
    legacy = Document(
        page_content="legacy",
        metadata={"source": "old.pdf", "page": 0},
    )

    assert rag.format_context([legacy]).startswith("[Source 1: old.pdf, page 1]")
    assert rag.format_sources([legacy]) == ["old.pdf — page 1"]


def test_source_formatting_uses_indexed_filename_and_one_based_page() -> None:
    current = Document(
        page_content="[FIGURE]\ndiagram",
        metadata={
            "filename": "new.pdf",
            "page_start": 3,
            "page_end": 3,
            "content_type": "figure",
        },
    )

    assert rag.format_sources([current]) == ["new.pdf — page 3"]


def test_get_vector_db_loads_only_the_active_index_path() -> None:
    calls: dict[str, object] = {}

    class FakeIndex:
        def active_db_path(self) -> Path:
            return Path("/tmp/db/indexes/verified")

    def embeddings_factory(**kwargs):
        calls["embedding"] = kwargs
        return "embedding"

    def vector_store_factory(**kwargs):
        calls["store"] = kwargs
        return "vector-store"

    result = rag.get_vector_db(
        index_manager=FakeIndex(),
        embeddings_factory=embeddings_factory,
        vector_store_factory=vector_store_factory,
    )

    assert result == "vector-store"
    assert calls["store"]["persist_directory"] == "/tmp/db/indexes/verified"
    assert calls["store"]["embedding_function"] == "embedding"


def test_get_vector_db_fails_clearly_without_an_active_pointer(tmp_path: Path) -> None:
    with pytest.raises(rag.ActiveIndexUnavailableError, match="No verified active index"):
        rag.resolve_active_index_path(
            active_index_file=tmp_path / "active_index.json",
            db_dir=tmp_path,
        )


def test_active_pointer_does_not_treat_an_unbuilt_empty_version_as_verified(
    tmp_path: Path,
) -> None:
    version = tmp_path / "indexes" / "initial-version"
    version.mkdir(parents=True)
    (version / ".DS_Store").write_text("not verification", encoding="utf-8")
    pointer = tmp_path / "active_index.json"
    pointer.write_text('{"version_id":"initial-version"}\n', encoding="utf-8")

    with pytest.raises(rag.ActiveIndexUnavailableError, match="No verified active index"):
        rag.resolve_active_index_path(active_index_file=pointer, db_dir=tmp_path)


def test_active_pointer_requires_index_manager_verification_marker(tmp_path: Path) -> None:
    version = tmp_path / "indexes" / "verified-version"
    version.mkdir(parents=True)
    (version / ".verified-index.json").write_text(
        '{"document_ids":["sha256:guide"],"chunk_count":2}\n', encoding="utf-8"
    )
    pointer = tmp_path / "active_index.json"
    pointer.write_text('{"version_id":"verified-version"}\n', encoding="utf-8")

    assert rag.resolve_active_index_path(
        active_index_file=pointer, db_dir=tmp_path
    ) == version.resolve()


def test_retrieval_rejects_remote_embedding_endpoint_before_factory_call(
    monkeypatch,
) -> None:
    calls: list[dict] = []

    class FakeIndex:
        def active_db_path(self) -> Path:
            return Path("/tmp/db/indexes/verified")

    monkeypatch.setattr(rag, "OLLAMA_BASE_URL", "https://example.com")

    with pytest.raises(ValueError, match="local loopback"):
        rag.get_vector_db(
            index_manager=FakeIndex(),
            embeddings_factory=lambda **kwargs: calls.append(kwargs),
            vector_store_factory=lambda **kwargs: kwargs,
        )

    assert calls == []


def test_retrieval_returns_enriched_source_records_without_model_calls(monkeypatch) -> None:
    doc = Document(
        page_content="[TABLE]\n| A | B |",
        metadata={
            "filename": "guide.pdf",
            "page_start": 4,
            "page_end": 5,
            "content_type": "table",
            "section_path": '["Configuration","Ports"]',
            "extraction_method": "docling",
        },
    )

    class FakeVectorDB:
        def similarity_search_with_relevance_scores(self, *, query: str, k: int):
            assert query == "table ports"
            assert k > 0
            return [(doc, 0.9)]

    monkeypatch.setattr(tools, "get_vector_db", lambda: FakeVectorDB())
    monkeypatch.setattr(tools, "expand_sdk_query", lambda question: question)

    docs, source_records = tools.retrieve_docs_tool("table ports")

    assert docs == [doc]
    assert source_records == [
        {
            "source": "guide.pdf",
            "page": None,
            "page_start": 4,
            "page_end": 5,
            "content_type": "table",
            "section_path": ["Configuration", "Ports"],
            "extraction_method": "docling",
            "score": 0.9,
            "passed_threshold": True,
        }
    ]
