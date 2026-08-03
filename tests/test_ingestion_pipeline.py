from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from document_models import ContentBlock, ContentType, IndexChunk, NormalizedDocument
from document_store import DocumentRecord
from ingest import IngestionError, IngestionPipeline, parse_args
from index_manager import IndexCleanupError


@dataclass
class FakeJob:
    job_id: str
    document_id: str
    state: str
    error: str | None = None


class FakeRegistry:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.jobs: list[FakeJob] = []
        self.fail_active = False

    @property
    def latest_job(self) -> FakeJob:
        return self.jobs[-1]

    def create_job(self, document_id: str, *, uploader: str | None = None) -> FakeJob:
        self.events.append("job:pending")
        job = FakeJob(f"job-{len(self.jobs) + 1}", document_id, "pending")
        self.jobs.append(job)
        return job

    def transition_job(
        self, job_id: str, state: str, *, error: str | None = None
    ) -> FakeJob:
        job = next(job for job in self.jobs if job.job_id == job_id)
        if state == "active" and self.fail_active:
            raise RuntimeError("registry unavailable")
        job.state = state
        job.error = error
        self.events.append(f"job:{state}")
        return job


class FakeStore:
    def __init__(self, events: list[str], paths: list[Path] | None = None) -> None:
        self.events = events
        self.paths = paths or [Path("guide.pdf")]
        self.failed: list[tuple[str, str]] = []
        self.active_records: dict[str, DocumentRecord] = {}

    def validate_pdf(self, path: Path) -> DocumentRecord:
        self.events.append(f"validate:{Path(path).name}")
        stem = Path(path).stem
        return DocumentRecord(
            document_id=f"sha256:{stem}",
            filename=Path(path).name,
            normalized_filename=Path(path).name.casefold(),
            sha256=stem,
            path=Path(path),
            size_bytes=100,
            page_count=2,
        )

    def register(self, record: DocumentRecord) -> DocumentRecord:
        self.events.append(f"register:{record.filename}")
        self.active_records[record.document_id] = record
        return record

    def mark_failed(self, document_id: str, error: str) -> None:
        self.events.append("document:failed")
        self.failed.append((document_id, error))
        self.active_records.pop(document_id, None)

    def list_active(self) -> list[DocumentRecord]:
        return list(self.active_records.values())

    def discover_corpus(self) -> list[Path]:
        self.events.append("discover")
        return list(self.paths)


class FakeExtractor:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def extract(self, path: Path, document_id: str) -> NormalizedDocument:
        self.events.append(f"extract:{Path(path).name}")
        return NormalizedDocument(
            document_id=document_id,
            filename=Path(path).name,
            blocks=(
                ContentBlock(
                    "table",
                    ContentType.TABLE,
                    "| A | B |",
                    1,
                    2,
                ),
                ContentBlock(
                    "ocr",
                    ContentType.TEXT,
                    "screen label",
                    2,
                    2,
                    metadata={"label": "ocr"},
                    extraction_method="ocr",
                ),
            ),
        )


class FakeEnricher:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def enrich(self, document: NormalizedDocument, path: Path) -> NormalizedDocument:
        self.events.append(f"enrich:{Path(path).name}")
        figure = ContentBlock(
            "figure",
            ContentType.FIGURE,
            "wiring diagram",
            2,
            2,
            extraction_method="ollama_vision",
        )
        return NormalizedDocument(
            document.document_id,
            document.filename,
            (*document.blocks, figure),
            ("one warning",),
        )


class FakeChunker:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def build(self, document: NormalizedDocument) -> list[IndexChunk]:
        self.events.append(f"chunk:{document.filename}")
        return [
            IndexChunk(
                chunk_id=f"chunk:{document.document_id}",
                document_id=document.document_id,
                filename=document.filename,
                content_type=ContentType.TABLE,
                text="[TABLE]\n| A | B |",
                page_start=1,
                page_end=2,
            )
        ]


class FakeIndex:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.fail = False
        self.cleanup_fail = False
        self.ids: set[str] = set()
        self.rebuilt: list[list[IndexChunk]] | None = None

    def upsert_document(self, chunks: list[IndexChunk]) -> None:
        self.events.append("index:upsert")
        self.ids.update(chunk.document_id for chunk in chunks)
        if self.fail:
            raise RuntimeError("index write failed")
        if self.cleanup_fail:
            raise IndexCleanupError(
                "retired index cleanup failed after active pointer switch",
                stale_path=Path("db/indexes/old-version"),
            )

    def delete_document(self, document_id: str) -> None:
        self.events.append("index:delete")
        self.ids.discard(document_id)

    def rebuild(
        self,
        documents: list[list[IndexChunk]],
        *,
        expected_document_ids: set[str],
        expected_chunk_count: int,
    ) -> Path:
        self.events.append("index:rebuild")
        self.rebuilt = documents
        assert {chunks[0].document_id for chunks in documents} == expected_document_ids
        assert sum(len(chunks) for chunks in documents) == expected_chunk_count
        self.ids = set(expected_document_ids)
        if self.cleanup_fail:
            raise IndexCleanupError(
                "retired index cleanup failed after active pointer switch",
                stale_path=Path("db/indexes/old-version"),
            )
        return Path("db/indexes/rebuilt-version")

    def verify(self, expected_document_ids: set[str], expected_chunk_count: int) -> None:
        assert self.ids == expected_document_ids
        assert expected_chunk_count == len(self.rebuilt or [])

    def active_db_path(self) -> Path:
        return Path("db/indexes/active-version")

    def count(self, *, document_id: str | None = None) -> int:
        if document_id is None:
            return len(self.ids)
        return int(document_id in self.ids)


@pytest.fixture
def pipeline() -> IngestionPipeline:
    events: list[str] = []
    return IngestionPipeline(
        store=FakeStore(events),
        extractor=FakeExtractor(events),
        enricher=FakeEnricher(events),
        chunker=FakeChunker(events),
        registry=FakeRegistry(events),
        index=FakeIndex(events),
        timer=lambda: 1.0,
    )


def test_ingest_pdf_runs_stages_in_order_and_reports_multimodal_counts(
    pipeline: IngestionPipeline,
) -> None:
    receipt = pipeline.ingest_pdf(Path("guide.pdf"), uploader="local-user")

    assert pipeline.registry.events == [
        "validate:guide.pdf",
        "register:guide.pdf",
        "job:pending",
        "job:extracting",
        "extract:guide.pdf",
        "enrich:guide.pdf",
        "chunk:guide.pdf",
        "job:indexing",
        "index:upsert",
        "job:active",
    ]
    assert receipt.filename == "guide.pdf"
    assert receipt.pages == 2
    assert receipt.tables == 1
    assert receipt.figures == 1
    assert receipt.ocr_blocks == 1
    assert receipt.chunks == 1
    assert receipt.warnings == ("one warning",)
    assert receipt.index_version == "active-version"
    assert receipt.uploader == "local-user"


def test_failed_ingestion_marks_job_failed_and_removes_partial_chunks(
    pipeline: IngestionPipeline,
) -> None:
    pipeline.index.fail = True

    with pytest.raises(IngestionError, match="indexing"):
        pipeline.ingest_pdf(Path("guide.pdf"))

    assert pipeline.registry.latest_job.state == "failed"
    assert pipeline.index.count(document_id="sha256:guide") == 0
    assert pipeline.store.failed[0][0] == "sha256:guide"
    assert pipeline.registry.events[-3:] == [
        "index:delete",
        "document:failed",
        "job:failed",
    ]


def test_post_switch_retirement_failure_keeps_active_chunks_and_becomes_warning(
    pipeline: IngestionPipeline,
) -> None:
    pipeline.index.cleanup_fail = True

    receipt = pipeline.ingest_pdf(Path("guide.pdf"))

    assert pipeline.registry.latest_job.state == "active"
    assert pipeline.index.count(document_id="sha256:guide") == 1
    assert any("retired index cleanup failed" in warning for warning in receipt.warnings)
    assert "index:delete" not in pipeline.registry.events


def test_failed_reingestion_preserves_previously_active_document_registry_state(
    pipeline: IngestionPipeline,
) -> None:
    existing = pipeline.store.validate_pdf(Path("guide.pdf"))
    pipeline.store.active_records[existing.document_id] = existing
    pipeline.index.ids.add(existing.document_id)
    pipeline.index.fail = True

    with pytest.raises(IngestionError):
        pipeline.ingest_pdf(Path("guide.pdf"))

    assert pipeline.store.failed == []
    assert pipeline.store.list_active() == [existing]


def test_post_switch_registry_failure_reports_reconciliation_without_deleting_index(
    pipeline: IngestionPipeline,
) -> None:
    pipeline.registry.fail_active = True

    with pytest.raises(IngestionError, match="registry_reconciliation"):
        pipeline.ingest_pdf(Path("guide.pdf"))

    assert pipeline.index.count(document_id="sha256:guide") == 1
    assert pipeline.registry.latest_job.state == "indexing"
    assert pipeline.store.failed == []
    assert "index:delete" not in pipeline.registry.events


def test_rebuild_corpus_processes_every_discovered_root_and_managed_pdf() -> None:
    events: list[str] = []
    paths = [Path("docs/root.pdf"), Path("docs/managed/upload.pdf")]
    pipeline = IngestionPipeline(
        store=FakeStore(events, paths),
        extractor=FakeExtractor(events),
        enricher=None,
        chunker=FakeChunker(events),
        registry=FakeRegistry(events),
        index=FakeIndex(events),
        timer=lambda: 2.0,
    )

    result = pipeline.rebuild_corpus()

    assert [receipt.filename for receipt in result.documents] == ["root.pdf", "upload.pdf"]
    assert pipeline.index.rebuilt is not None
    assert [chunks[0].document_id for chunks in pipeline.index.rebuilt] == [
        "sha256:root",
        "sha256:upload",
    ]
    assert "enrich:root.pdf" not in events
    assert result.index_version == "rebuilt-version"


def test_rebuild_failure_marks_the_current_extraction_job_failed() -> None:
    events: list[str] = []

    class FailingExtractor(FakeExtractor):
        def extract(self, path: Path, document_id: str) -> NormalizedDocument:
            super().extract(path, document_id)
            raise RuntimeError("conversion failed")

    registry = FakeRegistry(events)
    pipeline = IngestionPipeline(
        store=FakeStore(events),
        extractor=FailingExtractor(events),
        enricher=None,
        chunker=FakeChunker(events),
        registry=registry,
        index=FakeIndex(events),
        timer=lambda: 2.0,
    )

    with pytest.raises(IngestionError, match="rebuilding"):
        pipeline.rebuild_corpus()

    assert registry.latest_job.state == "failed"
    assert pipeline.store.failed[0][0] == "sha256:guide"


def test_rebuild_retirement_cleanup_failure_keeps_verified_version_active() -> None:
    events: list[str] = []
    registry = FakeRegistry(events)
    index = FakeIndex(events)
    index.cleanup_fail = True
    pipeline = IngestionPipeline(
        store=FakeStore(events),
        extractor=FakeExtractor(events),
        enricher=None,
        chunker=FakeChunker(events),
        registry=registry,
        index=index,
        timer=lambda: 2.0,
    )

    result = pipeline.rebuild_corpus()

    assert registry.latest_job.state == "active"
    assert result.index_version == "active-version"
    assert any(
        "retired index cleanup failed" in warning
        for warning in result.documents[0].warnings
    )


def test_receipt_uses_validated_page_count_and_counts_native_images() -> None:
    events: list[str] = []

    class FivePageStore(FakeStore):
        def validate_pdf(self, path: Path) -> DocumentRecord:
            record = super().validate_pdf(path)
            return DocumentRecord(
                record.document_id,
                record.filename,
                record.normalized_filename,
                record.sha256,
                record.path,
                record.size_bytes,
                5,
            )

    class NativeImageExtractor(FakeExtractor):
        def extract(self, path: Path, document_id: str) -> NormalizedDocument:
            document = super().extract(path, document_id)
            image = ContentBlock("image", ContentType.IMAGE, "chart", 1, 1)
            return NormalizedDocument(
                document.document_id,
                document.filename,
                (*document.blocks, image),
            )

    pipeline = IngestionPipeline(
        store=FivePageStore(events),
        extractor=NativeImageExtractor(events),
        enricher=None,
        chunker=FakeChunker(events),
        registry=FakeRegistry(events),
        index=FakeIndex(events),
        timer=lambda: 2.0,
    )

    receipt = pipeline.ingest_pdf(Path("guide.pdf"))

    assert receipt.pages == 5
    assert receipt.figures == 1


def test_rebuild_document_elapsed_time_stops_after_each_document_processing() -> None:
    events: list[str] = []
    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    pipeline = IngestionPipeline(
        store=FakeStore(events, [Path("one.pdf"), Path("two.pdf")]),
        extractor=FakeExtractor(events),
        enricher=None,
        chunker=FakeChunker(events),
        registry=FakeRegistry(events),
        index=FakeIndex(events),
        timer=lambda: next(ticks),
    )

    result = pipeline.rebuild_corpus()

    assert [receipt.elapsed_ms for receipt in result.documents] == [1000, 1000]


def test_cli_supports_incremental_add_and_deterministic_no_vision() -> None:
    args = parse_args(["--add", "guide.pdf", "--no-vision"])

    assert args.add == Path("guide.pdf")
    assert args.no_vision is True
    assert args.reset is False
