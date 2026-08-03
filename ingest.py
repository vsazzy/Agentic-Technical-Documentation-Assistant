"""Composition root and CLI for local multimodal PDF ingestion."""

from __future__ import annotations

import argparse
import sqlite3
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from config import CHUNK_OVERLAP, CHUNK_SIZE
from document_chunker import DocumentChunker
from document_models import ContentBlock, ContentType, IndexChunk, NormalizedDocument
from document_store import DocumentRecord, DocumentStore
from index_manager import IndexCleanupError, IndexManager
from pdf_extractor import DoclingBackend, PdfExtractor
from vision_enrichment import VisionClient, VisionEnricher


class IngestionError(RuntimeError):
    """A safe, stage-specific ingestion failure."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        document_id: str | None = None,
        job_id: str | None = None,
    ) -> None:
        self.stage = stage
        self.document_id = document_id
        self.job_id = job_id
        super().__init__(f"{stage}: {message}")


@dataclass(frozen=True)
class IngestionJob:
    job_id: str
    document_id: str
    state: str
    error: str | None = None


@dataclass(frozen=True)
class IngestionReceipt:
    job_id: str
    document_id: str
    filename: str
    pages: int
    tables: int
    figures: int
    ocr_blocks: int
    chunks: int
    warnings: tuple[str, ...]
    elapsed_ms: int
    index_version: str
    uploader: str | None = None


@dataclass(frozen=True)
class RebuildReceipt:
    documents: tuple[IngestionReceipt, ...]
    index_version: str
    elapsed_ms: int


class JobRegistry(Protocol):
    def create_job(
        self, document_id: str, *, uploader: str | None = None
    ) -> IngestionJob: ...

    def transition_job(
        self, job_id: str, state: str, *, error: str | None = None
    ) -> IngestionJob: ...


class SqliteIngestionRegistry:
    """Persist job transitions in the registry schema owned by DocumentStore."""

    _STATES = frozenset(
        {"pending", "extracting", "indexing", "active", "failed", "rebuilding"}
    )

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    @staticmethod
    def _from_row(row: sqlite3.Row) -> IngestionJob:
        return IngestionJob(
            job_id=str(row["job_id"]),
            document_id=str(row["document_id"]),
            state=str(row["status"]),
            error=row["error"],
        )

    def create_job(
        self, document_id: str, *, uploader: str | None = None
    ) -> IngestionJob:
        del uploader  # Reserved for the Slack lifecycle schema extension.
        job = IngestionJob(uuid.uuid4().hex, document_id, "pending")
        with self._store._connection() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_jobs (job_id, document_id, status, error)
                VALUES (?, ?, ?, NULL)
                """,
                (job.job_id, job.document_id, job.state),
            )
        return job

    def transition_job(
        self, job_id: str, state: str, *, error: str | None = None
    ) -> IngestionJob:
        if state not in self._STATES:
            raise ValueError(f"unsupported ingestion job state: {state}")
        with self._store._connection() as connection:
            result = connection.execute(
                "UPDATE ingestion_jobs SET status = ?, error = ? WHERE job_id = ?",
                (state, error, job_id),
            )
            if result.rowcount != 1:
                raise KeyError(f"unknown ingestion job: {job_id}")
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        assert row is not None
        return self._from_row(row)


class Extractor(Protocol):
    def extract(self, path: Path, document_id: str) -> NormalizedDocument: ...


class Enricher(Protocol):
    def enrich(
        self, document: NormalizedDocument, pdf_path: Path
    ) -> NormalizedDocument: ...


class Chunker(Protocol):
    def build(self, document: NormalizedDocument) -> list[IndexChunk]: ...


class IngestionPipeline:
    """Coordinate validation, extraction, enrichment, chunking, and indexing."""

    def __init__(
        self,
        *,
        store: Any,
        extractor: Extractor,
        enricher: Enricher | None,
        chunker: Chunker,
        registry: JobRegistry,
        index: Any,
        timer: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store = store
        self.extractor = extractor
        self.enricher = enricher
        self.chunker = chunker
        self.registry = registry
        self.index = index
        self._timer = timer

    @staticmethod
    def _job_id(job: IngestionJob | Any) -> str:
        job_id = getattr(job, "job_id", None)
        if not isinstance(job_id, str) or not job_id:
            raise TypeError("registry create_job must return an object with a job_id")
        return job_id

    def _extract_and_chunk(
        self, record: DocumentRecord, job_id: str
    ) -> tuple[NormalizedDocument, list[IndexChunk]]:
        self.registry.transition_job(job_id, "extracting")
        document = self.extractor.extract(record.path, record.document_id)
        if self.enricher is not None:
            document = self.enricher.enrich(document, record.path)
        chunks = self.chunker.build(document)
        if not chunks:
            raise ValueError("document produced no indexable chunks")
        return document, chunks

    @staticmethod
    def _is_ocr(block: ContentBlock) -> bool:
        label = block.metadata.get("label")
        return (
            isinstance(label, str)
            and label.casefold() in {"ocr", "ocr_text", "handwritten_text"}
        ) or "ocr" in block.extraction_method.casefold()

    def _receipt(
        self,
        *,
        job_id: str,
        document: NormalizedDocument,
        chunks: Sequence[IndexChunk],
        started_at: float,
        index_version: str,
        uploader: str | None,
        page_count: int,
        elapsed_ms: int | None = None,
    ) -> IngestionReceipt:
        return IngestionReceipt(
            job_id=job_id,
            document_id=document.document_id,
            filename=document.filename,
            pages=page_count,
            tables=sum(
                block.content_type is ContentType.TABLE for block in document.blocks
            ),
            figures=sum(
                block.content_type in {ContentType.IMAGE, ContentType.FIGURE}
                for block in document.blocks
            ),
            ocr_blocks=sum(self._is_ocr(block) for block in document.blocks),
            chunks=len(chunks),
            warnings=document.warnings,
            elapsed_ms=(
                max(0, round((self._timer() - started_at) * 1000))
                if elapsed_ms is None
                else elapsed_ms
            ),
            index_version=index_version,
            uploader=uploader,
        )

    @staticmethod
    def _error_detail(error: Exception) -> str:
        detail = " ".join(str(error).split())
        return detail or type(error).__name__

    def _mark_failed(
        self,
        *,
        record: DocumentRecord | None,
        job_id: str | None,
        error: Exception,
        mark_document: bool = True,
    ) -> None:
        detail = self._error_detail(error)
        registry_error: Exception | None = None
        if record is not None and mark_document:
            try:
                self.store.mark_failed(record.document_id, detail)
            except Exception as failure:
                registry_error = failure
        if job_id is not None:
            try:
                self.registry.transition_job(job_id, "failed", error=detail)
            except Exception as failure:
                registry_error = registry_error or failure
        if registry_error is not None:
            error.add_note(
                "failed to persist complete ingestion rollback state: "
                f"{self._error_detail(registry_error)}"
            )

    def _active_document_ids(self) -> set[str]:
        list_active = getattr(self.store, "list_active", None)
        if not callable(list_active):
            return set()
        return {
            record.document_id
            for record in list_active()
            if isinstance(getattr(record, "document_id", None), str)
        }

    def ingest_pdf(
        self, path: Path, uploader: str | None = None
    ) -> IngestionReceipt:
        """Incrementally ingest one validated corpus PDF."""
        started_at = self._timer()
        stage = "validation"
        record: DocumentRecord | None = None
        job_id: str | None = None
        index_touched = False
        had_existing_chunks = False
        was_active = False
        index_activated = False
        try:
            record = self.store.validate_pdf(Path(path))
            was_active = record.document_id in self._active_document_ids()
            record = self.store.register(record)
            job_id = self._job_id(
                self.registry.create_job(record.document_id, uploader=uploader)
            )
            stage = "extraction"
            document, chunks = self._extract_and_chunk(record, job_id)
            stage = "indexing"
            self.registry.transition_job(job_id, "indexing")
            had_existing_chunks = self.index.count(document_id=record.document_id) > 0
            index_touched = True
            try:
                self.index.upsert_document(chunks)
            except IndexCleanupError as cleanup_error:
                active_chunk_count = self.index.count(document_id=record.document_id)
                if cleanup_error.resource_path is not None or active_chunk_count != len(chunks):
                    raise
                document = NormalizedDocument(
                    document.document_id,
                    document.filename,
                    document.blocks,
                    (
                        *document.warnings,
                        "index activated but retired version cleanup failed: "
                        f"{self._error_detail(cleanup_error)}",
                    ),
                )
            index_activated = True
            self.registry.transition_job(job_id, "active")
            index_version = self.index.active_db_path().name
            return self._receipt(
                job_id=job_id,
                document=document,
                chunks=chunks,
                started_at=started_at,
                index_version=index_version,
                uploader=uploader,
                page_count=record.page_count,
            )
        except Exception as error:
            if index_activated:
                raise IngestionError(
                    "registry_reconciliation",
                    "the verified index is active but job-state reconciliation failed: "
                    f"{self._error_detail(error)}",
                    document_id=record.document_id if record is not None else None,
                    job_id=job_id,
                ) from error
            if (
                index_touched
                and record is not None
                and not had_existing_chunks
            ):
                try:
                    self.index.delete_document(record.document_id)
                except Exception as cleanup_error:
                    error.add_note(
                        "failed to remove partial chunks: "
                        f"{self._error_detail(cleanup_error)}"
                    )
            self._mark_failed(
                record=record,
                job_id=job_id,
                error=error,
                mark_document=not was_active,
            )
            if isinstance(error, IngestionError):
                raise
            raise IngestionError(
                stage,
                self._error_detail(error),
                document_id=record.document_id if record is not None else None,
                job_id=job_id,
            ) from error

    def rebuild_corpus(self) -> RebuildReceipt:
        """Build and atomically activate a complete version from discovered PDFs."""
        started_at = self._timer()
        paths = self.store.discover_corpus()
        if not paths:
            raise IngestionError("discovery", "no PDF documents found in the corpus")

        prepared: list[
            tuple[
                DocumentRecord,
                str,
                NormalizedDocument,
                list[IndexChunk],
                float,
                int,
            ]
        ] = []
        started_jobs: list[tuple[DocumentRecord, str]] = []
        previously_active_ids = self._active_document_ids()
        index_activated = False
        try:
            for path in paths:
                document_started_at = self._timer()
                record = self.store.register(self.store.validate_pdf(path))
                job_id = self._job_id(self.registry.create_job(record.document_id))
                started_jobs.append((record, job_id))
                document, chunks = self._extract_and_chunk(record, job_id)
                self.registry.transition_job(job_id, "indexing")
                document_elapsed_ms = max(
                    0, round((self._timer() - document_started_at) * 1000)
                )
                prepared.append(
                    (
                        record,
                        job_id,
                        document,
                        chunks,
                        document_started_at,
                        document_elapsed_ms,
                    )
                )

            expected_document_ids = {
                record.document_id for record, _, _, _, _, _ in prepared
            }
            expected_chunk_count = sum(
                len(chunks) for _, _, _, chunks, _, _ in prepared
            )
            rebuild_warning: str | None = None
            try:
                version_path = self.index.rebuild(
                    [chunks for _, _, _, chunks, _, _ in prepared],
                    expected_document_ids=expected_document_ids,
                    expected_chunk_count=expected_chunk_count,
                )
            except IndexCleanupError as cleanup_error:
                if cleanup_error.resource_path is not None:
                    raise
                self.index.verify(expected_document_ids, expected_chunk_count)
                version_path = self.index.active_db_path()
                rebuild_warning = (
                    "index activated but retired version cleanup failed: "
                    f"{self._error_detail(cleanup_error)}"
                )
            index_activated = True
            index_version = Path(version_path).name
            receipts: list[IngestionReceipt] = []
            for (
                record,
                job_id,
                document,
                chunks,
                document_started_at,
                document_elapsed_ms,
            ) in prepared:
                self.registry.transition_job(job_id, "active")
                if rebuild_warning is not None:
                    document = NormalizedDocument(
                        document.document_id,
                        document.filename,
                        document.blocks,
                        (*document.warnings, rebuild_warning),
                    )
                receipts.append(
                    self._receipt(
                        job_id=job_id,
                        document=document,
                        chunks=chunks,
                        started_at=document_started_at,
                        index_version=index_version,
                        uploader=None,
                        page_count=record.page_count,
                        elapsed_ms=document_elapsed_ms,
                    )
                )
            return RebuildReceipt(
                documents=tuple(receipts),
                index_version=index_version,
                elapsed_ms=max(0, round((self._timer() - started_at) * 1000)),
            )
        except Exception as error:
            if not index_activated:
                for record, job_id in started_jobs:
                    try:
                        if record.document_id not in previously_active_ids:
                            self.store.mark_failed(
                                record.document_id, self._error_detail(error)
                            )
                        self.registry.transition_job(
                            job_id, "failed", error=self._error_detail(error)
                        )
                    except Exception as transition_error:
                        error.add_note(
                            "failed to persist rebuild job rollback state: "
                            f"{self._error_detail(transition_error)}"
                        )
            else:
                raise IngestionError(
                    "registry_reconciliation",
                    "the verified index is active but job-state reconciliation failed: "
                    f"{self._error_detail(error)}",
                ) from error
            if isinstance(error, IngestionError):
                raise
            raise IngestionError("rebuilding", self._error_detail(error)) from error


def build_ingestion_pipeline(
    *,
    no_vision: bool = False,
    store: Any | None = None,
    extractor: Extractor | None = None,
    enricher: Enricher | None = None,
    chunker: Chunker | None = None,
    registry: JobRegistry | None = None,
    index: Any | None = None,
    timer: Callable[[], float] = time.monotonic,
) -> IngestionPipeline:
    """Compose production dependencies while allowing deterministic fakes."""
    resolved_store = store if store is not None else DocumentStore()
    resolved_extractor = (
        extractor
        if extractor is not None
        else PdfExtractor(DoclingBackend(), validate_pdf=resolved_store.validate_pdf)
    )
    if no_vision:
        resolved_enricher = None
    elif enricher is not None:
        resolved_enricher = enricher
    else:
        resolved_enricher = VisionEnricher(VisionClient.local())
    resolved_chunker = (
        chunker
        if chunker is not None
        else DocumentChunker(max_chars=CHUNK_SIZE, overlap_chars=CHUNK_OVERLAP)
    )
    resolved_registry = (
        registry
        if registry is not None
        else SqliteIngestionRegistry(resolved_store)
    )
    resolved_index = index if index is not None else IndexManager()
    return IngestionPipeline(
        store=resolved_store,
        extractor=resolved_extractor,
        enricher=resolved_enricher,
        chunker=resolved_chunker,
        registry=resolved_registry,
        index=resolved_index,
        timer=timer,
    )


def ingest_pdf(
    path: Path,
    uploader: str | None = None,
    *,
    pipeline: IngestionPipeline | None = None,
    no_vision: bool = False,
) -> IngestionReceipt:
    """Public incremental-ingestion entry point used by local and Slack callers."""
    active_pipeline = pipeline or build_ingestion_pipeline(no_vision=no_vision)
    return active_pipeline.ingest_pdf(Path(path), uploader=uploader)


def rebuild_corpus(
    *,
    pipeline: IngestionPipeline | None = None,
    no_vision: bool = False,
) -> RebuildReceipt:
    """Public full-corpus rebuild entry point."""
    active_pipeline = pipeline or build_ingestion_pipeline(no_vision=no_vision)
    return active_pipeline.rebuild_corpus()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest local PDFs into a versioned multimodal Chroma index."
    )
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument(
        "--reset",
        action="store_true",
        help="Build and atomically activate a fresh index from the complete corpus.",
    )
    operation.add_argument(
        "--add",
        type=Path,
        metavar="PATH",
        help="Incrementally ingest one PDF already under the configured corpus roots.",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Skip local vision enrichment for deterministic fallback operation.",
    )
    return parser.parse_args(argv)


def _print_document_receipt(receipt: IngestionReceipt) -> None:
    warning_text = "; ".join(receipt.warnings) if receipt.warnings else "none"
    print(
        f"{receipt.filename}: pages={receipt.pages}, tables={receipt.tables}, "
        f"figures={receipt.figures}, ocr_blocks={receipt.ocr_blocks}, "
        f"chunks={receipt.chunks}, warnings={warning_text}, "
        f"elapsed_ms={receipt.elapsed_ms}, job_id={receipt.job_id}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    pipeline = build_ingestion_pipeline(no_vision=args.no_vision)
    if args.add is not None:
        receipt = pipeline.ingest_pdf(args.add)
        _print_document_receipt(receipt)
        print(f"Active index version: {receipt.index_version}")
        return

    result = pipeline.rebuild_corpus()
    for receipt in result.documents:
        _print_document_receipt(receipt)
    print(
        f"Rebuilt {len(result.documents)} document(s) in {result.elapsed_ms} ms. "
        f"Active index version: {result.index_version}"
    )


if __name__ == "__main__":
    main()
