"""Incremental and recoverable management for local Chroma index versions."""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from langchain_core.documents import Document

from config import (
    ACTIVE_INDEX_FILE,
    COLLECTION_NAME,
    DB_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)
from document_models import IndexChunk


class IndexBuildError(RuntimeError):
    """Raised when an index mutation or version build cannot complete safely."""


class IndexVerificationError(IndexBuildError):
    """Raised when an index does not contain the expected corpus."""


class IndexCleanupError(IndexBuildError):
    """Raised when a failed or retired index version cannot be fully cleaned."""

    def __init__(self, message: str, *, stale_path: Path | None = None) -> None:
        super().__init__(message)
        self.stale_path = stale_path


class VectorStore(Protocol):
    """The small vector-store surface needed by :class:`IndexManager`."""

    def add_documents(self, documents: list[Document], *, ids: list[str]) -> Any: ...

    def get(
        self,
        *,
        where: dict[str, object] | None = None,
        include: list[str] | None = None,
    ) -> Mapping[str, Sequence[object]]: ...

    def delete(self, *, ids: list[str]) -> Any: ...

    def close(self) -> None: ...


StoreFactory = Callable[[Path], VectorStore]
SnapshotIndex = Callable[[Path, Path], None]
RemoveIndex = Callable[[Path], None]


def _validate_local_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Ollama base URL must be a local loopback URL")
    normalized = base_url.strip()
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Ollama base URL must be a local loopback URL")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("Ollama base URL must be a local loopback URL") from error

    hostname = parsed.hostname.lower()
    if hostname != "localhost":
        try:
            if not ipaddress.ip_address(hostname).is_loopback:
                raise ValueError
        except ValueError as error:
            raise ValueError("Ollama base URL must be a local loopback URL") from error
    return normalized.rstrip("/")


class _ChromaStore:
    """Adapter that keeps LangChain/Chroma details outside the manager."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def add_documents(self, documents: list[Document], *, ids: list[str]) -> Any:
        return self._store.add_documents(documents=documents, ids=ids)

    def get(
        self,
        *,
        where: dict[str, object] | None = None,
        include: list[str] | None = None,
    ) -> Mapping[str, Sequence[object]]:
        arguments: dict[str, object] = {}
        if where is not None:
            arguments["where"] = where
        if include is not None:
            arguments["include"] = include
        return self._store.get(**arguments)

    def delete(self, *, ids: list[str]) -> Any:
        return self._store.delete(ids=ids)

    def close(self) -> None:
        if self._store is None:
            return
        store = self._store
        try:
            # LangChain does not expose the client, but Chroma's persistent
            # client has a public close operation. Closing it is required before
            # copying or deleting the SQLite-backed version directory.
            store._client.close()
        finally:
            self._store = None


class IndexManager:
    """Own incremental writes and atomic switching of versioned local indexes."""

    def __init__(
        self,
        *,
        db_dir: Path = DB_DIR,
        active_index_file: Path = ACTIVE_INDEX_FILE,
        store_factory: StoreFactory | None = None,
        snapshot_index: SnapshotIndex | None = None,
        remove_index: RemoveIndex | None = None,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.db_dir = Path(db_dir).resolve()
        self.indexes_dir = self.db_dir / "indexes"
        self.active_index_file = Path(active_index_file).resolve()
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.ollama_base_url = _validate_local_base_url(ollama_base_url)
        self._store_factory = store_factory or self._build_chroma_store
        self._snapshot_index = snapshot_index or shutil.copytree
        self._remove_index = remove_index or shutil.rmtree
        self._ensure_active_version()

    def _build_chroma_store(self, path: Path) -> VectorStore:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.ollama_base_url,
        )
        return _ChromaStore(
            Chroma(
                persist_directory=str(path),
                embedding_function=embeddings,
                collection_name=self.collection_name,
            )
        )

    def _new_version_path(self) -> Path:
        return self.indexes_dir / uuid.uuid4().hex

    def _ensure_active_version(self) -> None:
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.active_index_file.parent.mkdir(parents=True, exist_ok=True)
        if self.active_index_file.exists():
            self.active_db_path()
            return
        initial_path = self._new_version_path()
        initial_path.mkdir()
        try:
            self._write_active_pointer(initial_path)
        except Exception:
            shutil.rmtree(initial_path, ignore_errors=True)
            raise

    def _write_active_pointer(self, path: Path) -> None:
        version_path = self._require_version_path(path)
        payload = {
            "version_id": version_path.name,
            "path": str(version_path),
        }
        temporary = self.active_index_file.with_name(
            f".{self.active_index_file.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("x", encoding="utf-8") as pointer_file:
                json.dump(payload, pointer_file, sort_keys=True)
                pointer_file.write("\n")
                pointer_file.flush()
                os.fsync(pointer_file.fileno())
            os.replace(temporary, self.active_index_file)
        finally:
            temporary.unlink(missing_ok=True)

    def _require_version_path(self, path: Path) -> Path:
        candidate = Path(path).resolve()
        try:
            relative = candidate.relative_to(self.indexes_dir.resolve())
        except ValueError as error:
            raise IndexBuildError("index version path escapes the indexes directory") from error
        if len(relative.parts) != 1 or relative.parts[0] in {"", ".", ".."}:
            raise IndexBuildError("index version path must name one version directory")
        return candidate

    def active_db_path(self) -> Path:
        """Return the validated path selected by the active-index pointer."""
        try:
            payload = json.loads(self.active_index_file.read_text(encoding="utf-8"))
            version_id = payload["version_id"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise IndexBuildError("active index pointer is unreadable") from error
        if not isinstance(version_id, str) or not version_id:
            raise IndexBuildError("active index pointer has an invalid version ID")
        path = self._require_version_path(self.indexes_dir / version_id)
        if not path.is_dir():
            raise IndexBuildError("active index directory does not exist")
        return path

    @staticmethod
    def _metadata_value(value: object) -> bool | int | float | str:
        if value is None:
            return "null"
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Mapping):
            serializable = {
                str(key): IndexManager._json_value(item)
                for key, item in value.items()
            }
        else:
            serializable = IndexManager._json_value(value)
        return json.dumps(serializable, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _json_value(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): IndexManager._json_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [IndexManager._json_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(IndexManager._json_value(item) for item in value)
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    @classmethod
    def _to_document(cls, chunk: IndexChunk) -> Document:
        metadata = {
            str(key): cls._metadata_value(value)
            for key, value in chunk.metadata.items()
        }
        metadata.update(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "content_type": chunk.content_type.value,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "section_path": json.dumps(list(chunk.section_path), separators=(",", ":")),
                "extraction_method": chunk.extraction_method,
            }
        )
        return Document(page_content=chunk.text, metadata=metadata)

    @staticmethod
    def _validate_document_chunks(chunks: Iterable[IndexChunk]) -> list[IndexChunk]:
        materialized = list(chunks)
        if not materialized:
            raise ValueError("upsert_document requires at least one chunk")
        if any(not isinstance(chunk, IndexChunk) for chunk in materialized):
            raise TypeError("upsert_document accepts only IndexChunk objects")
        document_ids = {chunk.document_id for chunk in materialized}
        if len(document_ids) != 1:
            raise ValueError("upsert_document chunks must belong to one document")
        chunk_ids = [chunk.chunk_id for chunk in materialized]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("upsert_document chunk IDs must be unique")
        return materialized

    @staticmethod
    def _ids(result: Mapping[str, Sequence[object]]) -> list[str]:
        ids = result.get("ids", ())
        return [str(chunk_id) for chunk_id in ids]

    def upsert_document(self, chunks: Iterable[IndexChunk]) -> None:
        """Replace all chunks for one document using deterministic chunk IDs."""
        materialized = self._validate_document_chunks(chunks)
        document_id = materialized[0].document_id
        chunk_ids = [chunk.chunk_id for chunk in materialized]
        previous_path = self.active_db_path()
        active_result = self._read_store(previous_path)
        active_document_ids = self._document_ids(active_result)
        active_count = len(self._ids(active_result))
        existing_count = len(
            self._ids_for_document(active_result, document_id=document_id)
        )
        candidate_path = self._new_version_path()
        store: VectorStore | None = None
        switched = False
        try:
            self._snapshot_index(previous_path, candidate_path)
            store = self._store_factory(candidate_path)
            existing = self._ids(
                store.get(where={"document_id": document_id}, include=["metadatas"])
            )
            if existing:
                store.delete(ids=existing)
            store.add_documents(
                [self._to_document(chunk) for chunk in materialized],
                ids=chunk_ids,
            )
            self._verify_store(
                store,
                active_document_ids | {document_id},
                active_count - existing_count + len(materialized),
            )
            store.close()
            store = None
            self._write_active_pointer(candidate_path)
            switched = True
        except Exception as error:
            self._abort_candidate(
                candidate_path,
                store,
                error,
                operation_message=f"document upsert failed: {document_id}",
            )

        assert switched
        self._retire(previous_path)

    def delete_document(self, document_id: str) -> None:
        """Delete all active chunks selected by document metadata."""
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be non-empty")
        previous_path = self.active_db_path()
        active_result = self._read_store(previous_path)
        active_document_ids = self._document_ids(active_result)
        existing_count = len(
            self._ids_for_document(active_result, document_id=document_id)
        )
        if existing_count == 0:
            return
        candidate_path = self._new_version_path()
        store: VectorStore | None = None
        switched = False
        try:
            self._snapshot_index(previous_path, candidate_path)
            store = self._store_factory(candidate_path)
            existing = self._ids(
                store.get(where={"document_id": document_id}, include=["metadatas"])
            )
            if existing:
                store.delete(ids=existing)
            self._verify_store(
                store,
                active_document_ids - {document_id},
                len(self._ids(active_result)) - existing_count,
            )
            store.close()
            store = None
            self._write_active_pointer(candidate_path)
            switched = True
        except Exception as error:
            self._abort_candidate(
                candidate_path,
                store,
                error,
                operation_message=f"document deletion failed: {document_id}",
            )

        assert switched
        self._retire(previous_path)

    def _read_store(self, path: Path) -> Mapping[str, Sequence[object]]:
        store = self._store_factory(path)
        try:
            return store.get(include=["metadatas"])
        finally:
            store.close()

    @classmethod
    def _ids_for_document(
        cls,
        result: Mapping[str, Sequence[object]],
        *,
        document_id: str,
    ) -> list[str]:
        ids = cls._ids(result)
        metadatas = result.get("metadatas", ())
        return [
            chunk_id
            for chunk_id, metadata in zip(ids, metadatas, strict=True)
            if isinstance(metadata, Mapping)
            and metadata.get("document_id") == document_id
        ]

    @staticmethod
    def _document_ids(result: Mapping[str, Sequence[object]]) -> set[str]:
        metadatas = result.get("metadatas", ())
        return {
            str(metadata["document_id"])
            for metadata in metadatas
            if isinstance(metadata, Mapping) and "document_id" in metadata
        }

    def _retire(self, path: Path) -> None:
        try:
            self._remove_index(path)
        except Exception as error:
            raise IndexCleanupError(
                "retired index cleanup failed after active pointer switch",
                stale_path=path,
            ) from error

    def _abort_candidate(
        self,
        candidate_path: Path,
        store: VectorStore | None,
        operation_error: Exception,
        *,
        operation_message: str,
    ) -> None:
        close_error: Exception | None = None
        remove_error: Exception | None = None
        try:
            if store is not None:
                store.close()
        except Exception as error:
            close_error = error
        finally:
            try:
                if candidate_path.exists():
                    self._remove_index(candidate_path)
            except Exception as error:
                remove_error = error

        if close_error is not None or remove_error is not None:
            failures = []
            if close_error is not None:
                failures.append("store close")
            if remove_error is not None:
                failures.append("directory removal")
            raise IndexCleanupError(
                f"candidate cleanup failed ({' and '.join(failures)})",
                stale_path=candidate_path if remove_error is not None else None,
            ) from operation_error
        if isinstance(operation_error, IndexBuildError):
            raise operation_error
        raise IndexBuildError(operation_message) from operation_error

    @staticmethod
    def _flatten_documents(documents: Iterable[Iterable[IndexChunk]]) -> list[IndexChunk]:
        flattened: list[IndexChunk] = []
        for document_chunks in documents:
            materialized = list(document_chunks)
            if not materialized:
                continue
            flattened.extend(IndexManager._validate_document_chunks(materialized))
        chunk_ids = [chunk.chunk_id for chunk in flattened]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("rebuild chunk IDs must be unique across documents")
        return flattened

    def rebuild(
        self,
        documents: Iterable[Iterable[IndexChunk]],
        *,
        expected_document_ids: set[str] | None = None,
        expected_chunk_count: int | None = None,
    ) -> Path:
        """Build, verify, and atomically activate a fresh index version."""
        chunks = self._flatten_documents(documents)
        actual_document_ids = {chunk.document_id for chunk in chunks}
        expected_ids = (
            actual_document_ids
            if expected_document_ids is None
            else set(expected_document_ids)
        )
        expected_count = len(chunks) if expected_chunk_count is None else expected_chunk_count
        previous_path = self.active_db_path()
        candidate_path = self._new_version_path()
        candidate_path.mkdir(parents=False)
        store: VectorStore | None = None
        switched = False
        try:
            store = self._store_factory(candidate_path)
            if chunks:
                store.add_documents(
                    [self._to_document(chunk) for chunk in chunks],
                    ids=[chunk.chunk_id for chunk in chunks],
                )
            self._verify_store(store, expected_ids, expected_count)
            store.close()
            store = None
            self._write_active_pointer(candidate_path)
            switched = True
        except Exception as error:
            self._abort_candidate(
                candidate_path,
                store,
                error,
                operation_message="index rebuild failed",
            )

        assert switched
        if previous_path != candidate_path:
            self._retire(previous_path)
        return candidate_path

    def _verify_store(
        self,
        store: VectorStore,
        expected_document_ids: set[str],
        expected_chunk_count: int,
    ) -> None:
        result = store.get(include=["metadatas"])
        ids = self._ids(result)
        if len(ids) != expected_chunk_count:
            raise IndexVerificationError(
                f"index chunk count mismatch: expected {expected_chunk_count}, found {len(ids)}"
            )
        metadatas = result.get("metadatas", ())
        actual_document_ids = {
            str(metadata["document_id"])
            for metadata in metadatas
            if isinstance(metadata, Mapping) and "document_id" in metadata
        }
        if actual_document_ids != expected_document_ids:
            raise IndexVerificationError(
                "index document IDs mismatch: "
                f"expected {sorted(expected_document_ids)}, found {sorted(actual_document_ids)}"
            )

    def verify(self, expected_document_ids: set[str], expected_chunk_count: int) -> None:
        """Verify exact chunk count and document identity in the active version."""
        if expected_chunk_count < 0:
            raise ValueError("expected_chunk_count must not be negative")
        store = self._store_factory(self.active_db_path())
        try:
            self._verify_store(store, set(expected_document_ids), expected_chunk_count)
        finally:
            store.close()

    def count(self, *, document_id: str | None = None) -> int:
        """Return an active chunk count, optionally filtered by document."""
        store = self._store_factory(self.active_db_path())
        try:
            where = {"document_id": document_id} if document_id is not None else None
            return len(self._ids(store.get(where=where, include=["metadatas"])))
        finally:
            store.close()
