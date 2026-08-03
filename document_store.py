"""Safe local PDF staging, storage, and registry operations."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pymupdf

from config import (
    DOCS_DIR,
    MANAGED_DOCS_DIR,
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    REGISTRY_FILE,
    STAGING_DOCS_DIR,
)


class DocumentValidationError(ValueError):
    """Raised when a document does not satisfy storage safety requirements."""


class DocumentDuplicateError(ValueError):
    """Raised when a registry uniqueness constraint would be violated."""


@dataclass(frozen=True)
class DocumentRecord:
    """The persisted identity and processing state of one PDF document."""

    document_id: str
    filename: str
    normalized_filename: str
    sha256: str
    path: Path
    size_bytes: int
    page_count: int
    status: str = "active"
    error: str | None = None


class DocumentStore:
    """Owns PDFs under the configured document roots and their SQLite registry."""

    def __init__(
        self,
        *,
        docs_dir: Path = DOCS_DIR,
        managed_dir: Path = MANAGED_DOCS_DIR,
        staging_dir: Path = STAGING_DOCS_DIR,
        registry_file: Path = REGISTRY_FILE,
        max_pdf_bytes: int = MAX_PDF_BYTES,
        max_pdf_pages: int = MAX_PDF_PAGES,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.managed_dir = Path(managed_dir)
        self.staging_dir = Path(staging_dir)
        self.registry_file = Path(registry_file)
        self.max_pdf_bytes = max_pdf_bytes
        self.max_pdf_pages = max_pdf_pages

    @staticmethod
    def normalize_filename(filename: str) -> str:
        """Return one safe PDF basename or reject a name that cannot be contained."""
        if not isinstance(filename, str):
            raise DocumentValidationError("filename must be a string")
        normalized = unicodedata.normalize("NFC", filename).strip()
        if not normalized or normalized in {".", ".."}:
            raise DocumentValidationError("filename must be non-empty")
        if Path(normalized).is_absolute() or "/" in normalized or "\\" in normalized:
            raise DocumentValidationError("filename must not contain a path")
        if Path(normalized).suffix.casefold() != ".pdf":
            raise DocumentValidationError("filename must end in .pdf")
        stem = Path(normalized).stem.strip()
        if not stem:
            raise DocumentValidationError("filename must include a stem")
        return f"{stem}.pdf"

    @staticmethod
    def _normalized_filename_key(filename: str) -> str:
        return DocumentStore.normalize_filename(filename).casefold()

    def _ensure_directories(self) -> None:
        for directory in (self.docs_dir, self.managed_dir, self.staging_dir, self.registry_file.parent):
            directory.mkdir(parents=True, exist_ok=True)
            if directory.is_symlink():
                raise DocumentValidationError(f"storage directory cannot be a symlink: {directory}")

    @staticmethod
    def _is_contained(path: Path, root: Path) -> bool:
        try:
            return path.resolve(strict=False).is_relative_to(root.resolve(strict=False))
        except (OSError, RuntimeError):
            return False

    def _require_regular_contained_file(self, path: Path, root: Path) -> Path:
        candidate = Path(path)
        if not self._is_contained(candidate, root):
            raise DocumentValidationError("document path escapes its managed directory")
        try:
            relative = candidate.relative_to(root)
        except ValueError as error:
            raise DocumentValidationError("document path escapes its managed directory") from error
        for ancestor in (
            root,
            *[root.joinpath(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1)],
        ):
            if ancestor.is_symlink():
                raise DocumentValidationError("symlinks are not accepted as document targets")
        if candidate.is_symlink() or not candidate.is_file():
            raise DocumentValidationError("document must be a regular file")
        return candidate

    def _validate_pdf_bytes(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise DocumentValidationError("PDF data must be bytes")
        if len(data) > self.max_pdf_bytes:
            raise DocumentValidationError("PDF size exceeds configured limit")
        if not data.startswith(b"%PDF-"):
            raise DocumentValidationError("PDF signature is invalid")

    def validate_pdf(self, path: Path) -> DocumentRecord:
        """Validate a managed PDF and return the immutable metadata to register."""
        self._ensure_directories()
        allowed_root = self.staging_dir if self._is_contained(Path(path), self.staging_dir) else self.docs_dir
        pdf_path = self._require_regular_contained_file(path, allowed_root)
        data = pdf_path.read_bytes()
        self._validate_pdf_bytes(data)
        try:
            with pymupdf.open(pdf_path) as document:
                page_count = document.page_count
        except (pymupdf.FileDataError, RuntimeError, ValueError) as error:
            raise DocumentValidationError("PDF cannot be opened") from error
        if page_count > self.max_pdf_pages:
            raise DocumentValidationError("PDF page count exceeds configured limit")

        filename = self.normalize_filename(pdf_path.name)
        sha256 = hashlib.sha256(data).hexdigest()
        return DocumentRecord(
            document_id=f"sha256:{sha256}",
            filename=filename,
            normalized_filename=filename.casefold(),
            sha256=sha256,
            path=pdf_path.resolve(),
            size_bytes=len(data),
            page_count=page_count,
        )

    def stage_bytes(self, filename: str, data: bytes) -> Path:
        """Validate and atomically create a regular staged PDF using a safe basename."""
        safe_filename = self.normalize_filename(filename)
        self._validate_pdf_bytes(data)
        self._ensure_directories()
        staged_path = self.staging_dir / safe_filename
        if not self._is_contained(staged_path, self.staging_dir):
            raise DocumentValidationError("staged path escapes the staging directory")
        if staged_path.exists() or staged_path.is_symlink():
            raise DocumentValidationError("staged document already exists")

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(staged_path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as staged_file:
                staged_file.write(data)
            self.validate_pdf(staged_path)
        except Exception:
            staged_path.unlink(missing_ok=True)
            raise
        return staged_path

    def promote(self, staged_path: Path) -> Path:
        """Move a staged PDF into managed storage, retaining an overwritten backup."""
        self._ensure_directories()
        source = self._require_regular_contained_file(staged_path, self.staging_dir)
        self.validate_pdf(source)
        destination = self.managed_dir / self.normalize_filename(source.name)
        if not self._is_contained(destination, self.managed_dir):
            raise DocumentValidationError("managed path escapes the managed directory")
        if destination.is_symlink():
            raise DocumentValidationError("symlinks are not accepted as upload targets")

        backup = destination.with_name(f"{destination.name}.backup")
        moved_existing = False
        if destination.exists():
            if backup.is_symlink():
                raise DocumentValidationError("backup path cannot be a symlink")
            os.replace(destination, backup)
            moved_existing = True
        try:
            os.replace(source, destination)
        except Exception:
            if moved_existing and backup.exists():
                os.replace(backup, destination)
            raise
        return destination

    def restore_backup(self, path: Path) -> Path:
        """Restore a ``.backup`` file created by :meth:`promote` into managed storage."""
        self._ensure_directories()
        backup = self._require_regular_contained_file(path, self.managed_dir)
        suffix = ".backup"
        if not backup.name.endswith(suffix):
            raise DocumentValidationError("backup file must end in .backup")
        destination = backup.with_name(backup.name[: -len(suffix)])
        if destination.is_symlink():
            raise DocumentValidationError("symlinks are not accepted as upload targets")
        os.replace(backup, destination)
        return destination

    def discover_corpus(self) -> list[Path]:
        """Discover root and managed PDFs, deliberately excluding in-progress staging files."""
        discovered: list[Path] = []
        if self.docs_dir.is_dir():
            discovered.extend(
                sorted(
                    (
                        path
                        for path in self.docs_dir.iterdir()
                        if path.is_file() and not path.is_symlink() and path.suffix.casefold() == ".pdf"
                    ),
                    key=lambda path: path.name.casefold(),
                )
            )
        if self.managed_dir.is_dir() and not self.managed_dir.is_symlink():
            discovered.extend(
                sorted(
                    (
                        path
                        for path in self.managed_dir.iterdir()
                        if path.is_file() and not path.is_symlink() and path.suffix.casefold() == ".pdf"
                    ),
                    key=lambda path: path.name.casefold(),
                )
            )
        return discovered

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self._ensure_directories()
        connection = sqlite3.connect(self.registry_file)
        connection.row_factory = sqlite3.Row
        try:
            self._create_schema(connection)
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                normalized_filename TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                page_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                error TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS documents_active_normalized_filename
                ON documents(normalized_filename) WHERE status = 'active';
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                job_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(document_id) REFERENCES documents(document_id)
            );
            CREATE TABLE IF NOT EXISTS index_versions (
                version_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(document_id) REFERENCES documents(document_id)
            );
            """
        )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            normalized_filename=row["normalized_filename"],
            sha256=row["sha256"],
            path=Path(row["path"]),
            size_bytes=row["size_bytes"],
            page_count=row["page_count"],
            status=row["status"],
            error=row["error"],
        )

    def register(self, record: DocumentRecord) -> None:
        """Register a document using SQLite constraints as the final duplicate guard."""
        with self._connection() as connection:
            existing = connection.execute(
                """
                SELECT document_id, sha256, normalized_filename
                FROM documents
                WHERE document_id = ? OR sha256 = ?
                   OR (normalized_filename = ? AND status = 'active')
                """,
                (record.document_id, record.sha256, record.normalized_filename),
            ).fetchone()
            if existing is not None:
                if existing["sha256"] == record.sha256:
                    raise DocumentDuplicateError("duplicate sha256")
                if existing["document_id"] == record.document_id:
                    raise DocumentDuplicateError("duplicate document_id")
                raise DocumentDuplicateError("duplicate active filename")
            try:
                connection.execute(
                    """
                    INSERT INTO documents (
                        document_id, filename, normalized_filename, sha256, path,
                        size_bytes, page_count, status, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.document_id,
                        record.filename,
                        record.normalized_filename,
                        record.sha256,
                        str(record.path),
                        record.size_bytes,
                        record.page_count,
                        record.status,
                        record.error,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise DocumentDuplicateError("duplicate document registry record") from error

    def get_by_filename(self, name: str) -> DocumentRecord | None:
        normalized_name = self._normalized_filename_key(name)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM documents
                WHERE normalized_filename = ? AND status != 'deleted'
                ORDER BY rowid DESC LIMIT 1
                """,
                (normalized_name,),
            ).fetchone()
        return self._record_from_row(row) if row is not None else None

    def list_active(self) -> list[DocumentRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE status = 'active' ORDER BY filename COLLATE NOCASE"
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def mark_failed(self, document_id: str, error: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE documents SET status = 'failed', error = ? WHERE document_id = ?",
                (error, document_id),
            )

    def mark_deleted(self, document_id: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "UPDATE documents SET status = 'deleted' WHERE document_id = ?",
                (document_id,),
            )
