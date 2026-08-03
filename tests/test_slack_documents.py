from pathlib import Path
from types import SimpleNamespace

import pymupdf

from document_store import DocumentStore
from ingest import IngestionReceipt, RebuildReceipt
from slack_documents import SlackDocumentService, is_management_channel, parse_management_request


def _pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class HappyPipeline:
    def __init__(self, store):
        self.store = store

    def ingest_pdf(self, path, uploader=None):
        record = self.store.register(self.store.validate_pdf(path))
        return IngestionReceipt("job", record.document_id, record.filename, record.page_count, 0, 0, 0, 1, (), 1, "index", uploader)

    def rebuild_corpus(self):
        records = [self.store.register(self.store.validate_pdf(path)) for path in self.store.discover_corpus()]
        receipts = tuple(IngestionReceipt("job", record.document_id, record.filename, record.page_count, 0, 0, 0, 1, (), 1, "index") for record in records)
        return RebuildReceipt(receipts, "index", 1)


def _service(tmp_path: Path):
    store = DocumentStore(
        docs_dir=tmp_path / "docs",
        managed_dir=tmp_path / "docs" / "managed",
        staging_dir=tmp_path / "docs" / "staging",
        registry_file=tmp_path / "db" / "registry.sqlite3",
    )
    return SlackDocumentService(store=store, pipeline=HappyPipeline(store)), store


def test_management_channel_and_natural_language_add():
    assert is_management_channel("C123", "C123")
    assert not is_management_channel("C999", "C123")
    assert parse_management_request("please add this PDF", has_pdf=True) == ("add", None)


def test_add_list_confirm_and_remove_happy_path(tmp_path):
    service, store = _service(tmp_path)
    store.staging_dir.mkdir(parents=True)
    first_staged = store.staging_dir / "first.pdf"
    _pdf(first_staged, "first")
    first = service.add_staged(first_staged, uploader_id="U1")
    second_staged = store.staging_dir / "second.pdf"
    _pdf(second_staged, "second")
    service.add_staged(second_staged, uploader_id="U1")

    assert [record.filename for record in service.list_documents()] == ["first.pdf", "second.pdf"]
    pending = service.request_removal("first.pdf", requester_id="U1", channel_id="C1")
    rebuilt = service.confirm_removal(pending.token, requester_id="U1", channel_id="C1")

    assert first.filename == "first.pdf"
    assert not (store.managed_dir / "first.pdf").exists()
    assert [receipt.filename for receipt in rebuilt.documents] == ["second.pdf"]
    assert [record.filename for record in service.list_documents()] == ["second.pdf"]
