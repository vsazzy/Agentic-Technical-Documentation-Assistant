from dataclasses import replace

import pymupdf
import pytest

from document_store import (
    DocumentDuplicateError,
    DocumentStore,
    DocumentValidationError,
)


@pytest.fixture
def pdf_bytes():
    document = pymupdf.open()
    document.new_page()
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def store(tmp_path):
    return DocumentStore(
        docs_dir=tmp_path / "docs",
        managed_dir=tmp_path / "docs" / "managed",
        staging_dir=tmp_path / "docs" / "staging",
        registry_file=tmp_path / "db" / "registry.sqlite3",
    )


def test_stage_normalizes_filename_and_uses_contained_staging_path(store, pdf_bytes):
    staged_path = store.stage_bytes("  Guide.PDF  ", pdf_bytes)

    assert staged_path == store.staging_dir / "Guide.pdf"
    assert staged_path.read_bytes() == pdf_bytes


def test_stage_rejects_path_traversal(store):
    with pytest.raises(DocumentValidationError):
        store.stage_bytes("../../secret.pdf", b"%PDF-1.7\n")


@pytest.mark.parametrize("data", [b"not a PDF", b"\n%PDF-1.7\n"])
def test_stage_rejects_files_without_pdf_signature(store, data):
    with pytest.raises(DocumentValidationError, match="signature"):
        store.stage_bytes("guide.pdf", data)


def test_stage_enforces_size_limit(store, pdf_bytes):
    limited_store = DocumentStore(
        docs_dir=store.docs_dir,
        managed_dir=store.managed_dir,
        staging_dir=store.staging_dir,
        registry_file=store.registry_file,
        max_pdf_bytes=len(pdf_bytes) - 1,
    )

    with pytest.raises(DocumentValidationError, match="size"):
        limited_store.stage_bytes("guide.pdf", pdf_bytes)


def test_stage_enforces_page_limit(store):
    document = pymupdf.open()
    document.new_page()
    document.new_page()
    data = document.tobytes()
    document.close()
    limited_store = DocumentStore(
        docs_dir=store.docs_dir,
        managed_dir=store.managed_dir,
        staging_dir=store.staging_dir,
        registry_file=store.registry_file,
        max_pdf_pages=1,
    )

    with pytest.raises(DocumentValidationError, match="page"):
        limited_store.stage_bytes("guide.pdf", data)


def test_discover_corpus_excludes_staging(store, pdf_bytes):
    store.docs_dir.mkdir(parents=True)
    store.managed_dir.mkdir()
    store.staging_dir.mkdir()
    (store.docs_dir / "root.pdf").write_bytes(pdf_bytes)
    (store.managed_dir / "managed.pdf").write_bytes(pdf_bytes)
    (store.staging_dir / "partial.pdf").write_bytes(pdf_bytes)

    assert [path.name for path in store.discover_corpus()] == ["root.pdf", "managed.pdf"]


def test_promote_moves_valid_staged_pdf_into_managed_directory(store, pdf_bytes):
    staged_path = store.stage_bytes("guide.pdf", pdf_bytes)

    promoted_path = store.promote(staged_path)

    assert promoted_path == store.managed_dir / "guide.pdf"
    assert promoted_path.read_bytes() == pdf_bytes
    assert not staged_path.exists()


def test_restore_backup_restores_replaced_managed_document(store, pdf_bytes):
    managed_path = store.promote(store.stage_bytes("guide.pdf", pdf_bytes))
    replacement = pymupdf.open()
    replacement.new_page(width=200, height=200)
    replacement_bytes = replacement.tobytes()
    replacement.close()
    store.promote(store.stage_bytes("guide.pdf", replacement_bytes))

    restored_path = store.restore_backup(store.managed_dir / "guide.pdf.backup")

    assert restored_path == managed_path
    assert restored_path.read_bytes() == pdf_bytes
    assert not (store.managed_dir / "guide.pdf.backup").exists()


def test_register_prevents_duplicate_sha256_and_active_normalized_filename(store, pdf_bytes):
    first_path = store.promote(store.stage_bytes("Guide.pdf", pdf_bytes))
    first_record = store.validate_pdf(first_path)
    store.register(first_record)

    second_path = store.promote(store.stage_bytes("other.pdf", pdf_bytes))
    with pytest.raises(DocumentDuplicateError, match="sha256"):
        store.register(store.validate_pdf(second_path))

    different_pdf = pymupdf.open()
    different_pdf.new_page(width=200, height=200)
    different_bytes = different_pdf.tobytes()
    different_pdf.close()
    filename_collision = store.stage_bytes("guide.PDF", different_bytes)
    with pytest.raises(DocumentDuplicateError, match="filename"):
        store.register(store.validate_pdf(filename_collision))


def test_registry_retrieves_active_records_and_tracks_failure_and_deletion(store, pdf_bytes):
    promoted_path = store.promote(store.stage_bytes("guide.pdf", pdf_bytes))
    record = store.validate_pdf(promoted_path)
    store.register(record)

    assert store.get_by_filename("GUIDE.pdf") == record
    assert store.list_active() == [record]

    store.mark_failed(record.document_id, "extractor unavailable")
    failed_record = store.get_by_filename("guide.pdf")
    assert failed_record.error == "extractor unavailable"
    assert failed_record.status == "failed"
    assert store.list_active() == []

    store.mark_deleted(record.document_id)
    assert store.get_by_filename("guide.pdf") is None
    assert store.list_active() == []


def test_registry_enforces_document_id_uniqueness(store, pdf_bytes):
    promoted_path = store.promote(store.stage_bytes("guide.pdf", pdf_bytes))
    record = store.validate_pdf(promoted_path)
    store.register(record)

    duplicate_id = replace(
        record,
        filename="other.pdf",
        normalized_filename="other.pdf",
        sha256="different-sha256",
    )
    with pytest.raises(DocumentDuplicateError, match="document_id"):
        store.register(duplicate_id)
