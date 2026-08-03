from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

import index_manager
from document_models import ContentType, IndexChunk
from index_manager import IndexBuildError, IndexManager, IndexVerificationError


class FakeVectorStore:
    def __init__(self, path: Path, factory: "FakeStoreFactory") -> None:
        self.path = path
        self.factory = factory
        self.records: dict[str, object] = {}
        self.close_calls = 0
        self.is_open = False
        self.fail_on_close = False

    def add_documents(self, documents: list[object], *, ids: list[str]) -> None:
        if self.factory.fail_on_add:
            raise RuntimeError("embedding failed")
        if len(documents) != len(ids):
            raise AssertionError("documents and ids must stay aligned")
        for chunk_id, document in zip(ids, documents, strict=True):
            if chunk_id in self.records:
                raise AssertionError("duplicate IDs must be deleted before add")
            self.records[chunk_id] = document
            if self.factory.fail_after_first_add:
                raise RuntimeError("embedding batch failed midway")

    def get(
        self,
        *,
        where: dict[str, object] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list[object]]:
        if self.factory.before_get is not None:
            self.factory.before_get(self.path)
        matching = [
            (chunk_id, document)
            for chunk_id, document in self.records.items()
            if where is None
            or all(document.metadata.get(key) == value for key, value in where.items())
        ]
        return {
            "ids": [chunk_id for chunk_id, _ in matching],
            "metadatas": [document.metadata for _, document in matching],
        }

    def delete(self, *, ids: list[str]) -> None:
        for chunk_id in ids:
            self.records.pop(chunk_id, None)
            if self.factory.fail_after_first_delete:
                raise RuntimeError("delete failed midway")

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_on_close:
            raise RuntimeError("store close failed")
        self.is_open = False


class FakeStoreFactory:
    def __init__(self) -> None:
        self.stores: dict[Path, FakeVectorStore] = {}
        self.fail_on_add = False
        self.fail_after_first_add = False
        self.fail_after_first_delete = False
        self.fail_new_store_close = False
        self.fail_remove = False
        self.before_get: Callable[[Path], None] | None = None

    def __call__(self, path: Path) -> FakeVectorStore:
        normalized = Path(path)
        is_new = normalized not in self.stores
        store = self.stores.setdefault(normalized, FakeVectorStore(normalized, self))
        if is_new and self.fail_new_store_close:
            store.fail_on_close = True
        store.is_open = True
        return store

    def snapshot(self, source: Path, destination: Path) -> None:
        source_store = self.stores.get(Path(source))
        if source_store is not None and source_store.is_open:
            raise AssertionError("active store must be closed before snapshot")
        Path(destination).mkdir()
        candidate = FakeVectorStore(Path(destination), self)
        candidate.fail_on_close = self.fail_new_store_close
        if source_store is not None:
            candidate.records = dict(source_store.records)
        self.stores[Path(destination)] = candidate

    def remove(self, path: Path) -> None:
        if self.fail_remove:
            raise OSError("candidate removal failed")
        shutil.rmtree(path)


@pytest.fixture
def chunks() -> list[IndexChunk]:
    return [
        IndexChunk(
            chunk_id="chunk-a",
            document_id="sha256:guide",
            filename="guide.pdf",
            content_type=ContentType.TEXT,
            text="Install the SDK.",
            page_start=1,
            page_end=1,
            section_path=("Setup",),
            metadata={"block_ids": ["block-a"], "confidence": 0.9},
        ),
        IndexChunk(
            chunk_id="chunk-b",
            document_id="sha256:guide",
            filename="guide.pdf",
            content_type=ContentType.TABLE,
            text="| Option | Default |",
            page_start=2,
            page_end=3,
            section_path=("Reference", "Options"),
            metadata={"block_ids": ["block-b"]},
        ),
    ]


@pytest.fixture
def other_chunks() -> list[IndexChunk]:
    return [
        IndexChunk(
            chunk_id="chunk-c",
            document_id="sha256:api",
            filename="api.pdf",
            content_type=ContentType.FIGURE,
            text="Request flow diagram.",
            page_start=4,
            page_end=4,
            section_path=("API",),
            metadata={"block_ids": ["block-c"]},
            extraction_method="ollama_vision",
        )
    ]


@pytest.fixture
def replacement_chunks() -> list[IndexChunk]:
    return [
        IndexChunk(
            chunk_id="chunk-replacement-a",
            document_id="sha256:guide",
            filename="guide.pdf",
            content_type=ContentType.TEXT,
            text="Install the replacement SDK.",
            page_start=1,
            page_end=1,
            section_path=("Setup",),
        ),
        IndexChunk(
            chunk_id="chunk-replacement-b",
            document_id="sha256:guide",
            filename="guide.pdf",
            content_type=ContentType.TEXT,
            text="Configure the replacement SDK.",
            page_start=2,
            page_end=2,
            section_path=("Setup",),
        ),
    ]


@pytest.fixture
def fake_factory() -> FakeStoreFactory:
    return FakeStoreFactory()


@pytest.fixture
def manager(tmp_path: Path, fake_factory: FakeStoreFactory) -> IndexManager:
    return IndexManager(
        db_dir=tmp_path / "db",
        active_index_file=tmp_path / "db" / "active_index.json",
        store_factory=fake_factory,
        snapshot_index=fake_factory.snapshot,
        remove_index=fake_factory.remove,
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.com",
        "http://192.168.1.20:11434",
        "http://127.0.0.1.example.com:11434",
        "http://user@127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?target=remote",
        "http://127.0.0.1:11434#remote",
        "ftp://127.0.0.1:11434",
    ],
)
def test_remote_or_ambiguous_embedding_url_is_rejected(
    tmp_path: Path,
    fake_factory: FakeStoreFactory,
    base_url: str,
) -> None:
    with pytest.raises(ValueError, match="local loopback"):
        IndexManager(
            db_dir=tmp_path / "db",
            active_index_file=tmp_path / "db" / "active_index.json",
            store_factory=fake_factory,
            snapshot_index=fake_factory.snapshot,
            remove_index=fake_factory.remove,
            ollama_base_url=base_url,
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:11434",
        "https://localhost:11434/",
        "http://[::1]:11434",
    ],
)
def test_loopback_embedding_url_is_accepted(
    tmp_path: Path,
    fake_factory: FakeStoreFactory,
    base_url: str,
) -> None:
    manager = IndexManager(
        db_dir=tmp_path / "db",
        active_index_file=tmp_path / "db" / "active_index.json",
        store_factory=fake_factory,
        snapshot_index=fake_factory.snapshot,
        remove_index=fake_factory.remove,
        ollama_base_url=base_url,
    )

    assert manager.ollama_base_url == base_url.rstrip("/")


def test_chroma_adapter_closes_its_persistent_client() -> None:
    class FakeClient:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class FakeChroma:
        _client = FakeClient()

    adapter = index_manager._ChromaStore(FakeChroma())

    adapter.close()

    assert FakeChroma._client.close_calls == 1


def test_upsert_uses_explicit_stable_ids_and_complete_chunk_metadata(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)

    store = fake_factory.stores[manager.active_db_path()]
    assert set(store.records) == {"chunk-a", "chunk-b"}
    first = store.records["chunk-a"]
    assert first.page_content == "Install the SDK."
    assert first.metadata == {
        "block_ids": '["block-a"]',
        "confidence": 0.9,
        "chunk_id": "chunk-a",
        "document_id": "sha256:guide",
        "filename": "guide.pdf",
        "content_type": "text",
        "page_start": 1,
        "page_end": 1,
        "section_path": '["Setup"]',
        "extraction_method": "docling",
    }


def test_retry_does_not_duplicate_chunks(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)
    manager.upsert_document(chunks)

    store = fake_factory.stores[manager.active_db_path()]
    assert set(store.records) == {"chunk-a", "chunk-b"}
    assert manager.count(document_id="sha256:guide") == 2


def test_failed_incremental_add_removes_partially_inserted_chunks(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
) -> None:
    fake_factory.fail_after_first_add = True

    with pytest.raises(IndexBuildError, match="upsert failed"):
        manager.upsert_document(chunks)

    fake_factory.fail_after_first_add = False
    assert manager.count(document_id="sha256:guide") == 0


def test_failed_replacement_preserves_active_path_and_exact_previous_contents(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
    replacement_chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)
    old_path = manager.active_db_path()
    old_records = dict(fake_factory.stores[old_path].records)
    fake_factory.fail_after_first_add = True

    with pytest.raises(IndexBuildError, match="upsert failed"):
        manager.upsert_document(replacement_chunks)

    fake_factory.fail_after_first_add = False
    assert manager.active_db_path() == old_path
    assert old_path.is_dir()
    assert fake_factory.stores[old_path].records == old_records


def test_successful_incremental_upsert_switches_to_a_new_version(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
) -> None:
    old_path = manager.active_db_path()

    manager.upsert_document(chunks)

    new_path = manager.active_db_path()
    assert new_path != old_path
    assert set(fake_factory.stores[new_path].records) == {"chunk-a", "chunk-b"}
    assert not old_path.exists()


def test_delete_document_resolves_chunk_ids_by_document_metadata(
    manager: IndexManager,
    chunks: list[IndexChunk],
    other_chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)
    manager.upsert_document(other_chunks)

    manager.delete_document("sha256:guide")

    assert manager.count(document_id="sha256:guide") == 0
    assert manager.count(document_id="sha256:api") == 1


def test_successful_delete_switches_version_without_mutating_old_store(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
    other_chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)
    manager.upsert_document(other_chunks)
    old_path = manager.active_db_path()
    old_records = dict(fake_factory.stores[old_path].records)

    manager.delete_document("sha256:guide")

    new_path = manager.active_db_path()
    assert new_path != old_path
    assert set(fake_factory.stores[new_path].records) == {"chunk-c"}
    assert fake_factory.stores[old_path].records == old_records


def test_failed_delete_preserves_active_path_and_exact_previous_contents(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
    other_chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)
    manager.upsert_document(other_chunks)
    old_path = manager.active_db_path()
    old_records = dict(fake_factory.stores[old_path].records)
    fake_factory.fail_after_first_delete = True

    with pytest.raises(IndexBuildError, match="deletion failed"):
        manager.delete_document("sha256:guide")

    fake_factory.fail_after_first_delete = False
    assert manager.active_db_path() == old_path
    assert old_path.is_dir()
    assert fake_factory.stores[old_path].records == old_records


def test_failed_rebuild_keeps_active_pointer_and_old_index(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)
    old_path = manager.active_db_path()
    fake_factory.fail_on_add = True

    with pytest.raises(IndexBuildError, match="rebuild failed"):
        manager.rebuild([chunks])

    assert manager.active_db_path() == old_path
    assert old_path.is_dir()
    assert list(old_path.parent.iterdir()) == [old_path]


def test_failed_rebuild_closes_and_removes_only_candidate_version(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)
    old_path = manager.active_db_path()
    old_close_calls = fake_factory.stores[old_path].close_calls
    paths_before_rebuild = set(fake_factory.stores)
    fake_factory.fail_on_add = True

    with pytest.raises(IndexBuildError):
        manager.rebuild([chunks])

    candidate_paths = [
        path for path in fake_factory.stores if path not in paths_before_rebuild
    ]
    assert len(candidate_paths) == 1
    candidate_path = candidate_paths[0]
    assert fake_factory.stores[candidate_path].close_calls == 1
    assert not candidate_path.exists()
    assert fake_factory.stores[old_path].close_calls == old_close_calls


def test_candidate_close_failure_still_removes_candidate_and_preserves_pointer(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)
    old_path = manager.active_db_path()
    paths_before_rebuild = set(fake_factory.stores)
    fake_factory.fail_new_store_close = True

    with pytest.raises(IndexBuildError, match="cleanup") as raised:
        manager.rebuild([chunks])

    candidate_paths = set(fake_factory.stores) - paths_before_rebuild
    assert len(candidate_paths) == 1
    candidate_path = candidate_paths.pop()
    assert type(raised.value).__name__ == "IndexCleanupError"
    assert manager.active_db_path() == old_path
    assert not candidate_path.exists()


def test_candidate_removal_failure_is_reported_with_stale_path_and_stable_pointer(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)
    old_path = manager.active_db_path()
    paths_before_rebuild = set(fake_factory.stores)
    fake_factory.fail_on_add = True
    fake_factory.fail_remove = True

    with pytest.raises(IndexBuildError, match="cleanup") as raised:
        manager.rebuild([chunks])

    candidate_paths = set(fake_factory.stores) - paths_before_rebuild
    assert len(candidate_paths) == 1
    candidate_path = candidate_paths.pop()
    assert type(raised.value).__name__ == "IndexCleanupError"
    assert getattr(raised.value, "stale_path") == candidate_path
    assert manager.active_db_path() == old_path
    assert candidate_path.is_dir()


def test_unverified_rebuild_preserves_pointer_and_old_index(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)
    old_path = manager.active_db_path()

    with pytest.raises(IndexVerificationError, match="document IDs"):
        manager.rebuild([chunks], expected_document_ids={"sha256:missing"})

    assert manager.active_db_path() == old_path
    assert old_path.is_dir()


def test_verified_rebuild_switches_pointer_before_retiring_old_index(
    manager: IndexManager,
    fake_factory: FakeStoreFactory,
    chunks: list[IndexChunk],
    other_chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)
    old_path = manager.active_db_path()

    def assert_old_version_is_still_active_during_verification(candidate_path: Path) -> None:
        if candidate_path != old_path:
            assert manager.active_db_path() == old_path
            assert old_path.is_dir()

    fake_factory.before_get = assert_old_version_is_still_active_during_verification
    new_path = manager.rebuild([chunks, other_chunks])

    assert manager.active_db_path() == new_path
    assert set(fake_factory.stores[new_path].records) == {"chunk-a", "chunk-b", "chunk-c"}
    assert not old_path.exists()


def test_verify_checks_exact_document_set_and_chunk_count(
    manager: IndexManager,
    chunks: list[IndexChunk],
) -> None:
    manager.upsert_document(chunks)

    manager.verify({"sha256:guide"}, 2)
    with pytest.raises(IndexVerificationError, match="chunk count"):
        manager.verify({"sha256:guide"}, 3)
    with pytest.raises(IndexVerificationError, match="document IDs"):
        manager.verify({"sha256:other"}, 2)
