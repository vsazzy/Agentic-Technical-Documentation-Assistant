from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from document_models import ContentType, IndexChunk
from index_manager import IndexBuildError, IndexManager, IndexVerificationError


class FakeVectorStore:
    def __init__(self, path: Path, factory: "FakeStoreFactory") -> None:
        self.path = path
        self.factory = factory
        self.records: dict[str, object] = {}
        self.close_calls = 0

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

    def close(self) -> None:
        self.close_calls += 1


class FakeStoreFactory:
    def __init__(self) -> None:
        self.stores: dict[Path, FakeVectorStore] = {}
        self.fail_on_add = False
        self.fail_after_first_add = False
        self.before_get: Callable[[Path], None] | None = None

    def __call__(self, path: Path) -> FakeVectorStore:
        normalized = Path(path)
        return self.stores.setdefault(normalized, FakeVectorStore(normalized, self))


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
def fake_factory() -> FakeStoreFactory:
    return FakeStoreFactory()


@pytest.fixture
def manager(tmp_path: Path, fake_factory: FakeStoreFactory) -> IndexManager:
    return IndexManager(
        db_dir=tmp_path / "db",
        active_index_file=tmp_path / "db" / "active_index.json",
        store_factory=fake_factory,
    )


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
    fake_factory.fail_on_add = True

    with pytest.raises(IndexBuildError):
        manager.rebuild([chunks])

    candidate_paths = [path for path in fake_factory.stores if path != old_path]
    assert len(candidate_paths) == 1
    candidate_path = candidate_paths[0]
    assert fake_factory.stores[candidate_path].close_calls == 1
    assert not candidate_path.exists()
    assert fake_factory.stores[old_path].close_calls == old_close_calls


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
