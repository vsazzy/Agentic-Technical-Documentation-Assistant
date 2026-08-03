import pytest

from document_models import ContentBlock, ContentType, IndexChunk, NormalizedDocument


def test_content_block_requires_one_based_page():
    with pytest.raises(ValueError, match="page_start"):
        ContentBlock(
            block_id="b1",
            content_type=ContentType.TEXT,
            text="hello",
            page_start=0,
            page_end=1,
        )


def test_content_block_rejects_unknown_content_type():
    with pytest.raises(ValueError, match="content_type"):
        ContentBlock("b1", "unknown", "hello", 1, 1)


def test_normalized_document_counts_content_types():
    doc = NormalizedDocument(
        document_id="sha256:abc",
        filename="guide.pdf",
        blocks=[
            ContentBlock("b1", ContentType.TEXT, "intro", 1, 1),
            ContentBlock("b2", ContentType.TABLE, "|A|B|", 2, 2),
        ],
    )

    assert doc.stats.tables == 1
    assert doc.stats.pages == 2


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: NormalizedDocument("", "guide.pdf", []), "document_id"),
        (lambda: ContentBlock("", ContentType.TEXT, "hello", 1, 1), "block_id"),
        (lambda: ContentBlock("b1", ContentType.TEXT, "", 1, 1), "text"),
        (
            lambda: IndexChunk("", "sha256:abc", "guide.pdf", ContentType.TEXT, "hello", 1, 1),
            "chunk_id",
        ),
    ],
)
def test_contracts_reject_empty_identifiers_and_content(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


def test_serialization_is_deterministic():
    doc = NormalizedDocument(
        document_id="sha256:abc",
        filename="guide.pdf",
        blocks=[
            ContentBlock(
                "b1",
                ContentType.TEXT,
                "intro",
                1,
                1,
                section_path=("Introduction",),
                metadata={"z": 1, "a": 2},
            )
        ],
    )

    assert doc.to_json() == (
        '{"blocks":[{"block_id":"b1","content_type":"text","extraction_method":"docling",'
        '"metadata":{"a":2,"z":1},"page_end":1,"page_start":1,'
        '"section_path":["Introduction"],"text":"intro"}],'
        '"document_id":"sha256:abc","filename":"guide.pdf"}'
    )


def test_contract_collections_cannot_be_mutated_after_validation():
    source_metadata = {"source": "native"}
    block = ContentBlock("b1", ContentType.TEXT, "intro", 1, 1, metadata=source_metadata)
    source_blocks = [block]
    document = NormalizedDocument("sha256:abc", "guide.pdf", source_blocks)
    chunk = IndexChunk("c1", "sha256:abc", "guide.pdf", ContentType.TEXT, "intro", 1, 1)

    source_metadata["source"] = "changed"
    source_blocks.append(ContentBlock("b2", ContentType.TABLE, "|A|", 2, 2))

    assert document.blocks == (block,)
    assert document.blocks[0].metadata["source"] == "native"
    with pytest.raises(TypeError):
        block.metadata["source"] = "changed"
    with pytest.raises(TypeError):
        chunk.metadata["source"] = "changed"


def test_serialization_returns_copies_of_contract_metadata():
    block = ContentBlock("b1", ContentType.TEXT, "intro", 1, 1, metadata={"source": "native"})
    document = NormalizedDocument("sha256:abc", "guide.pdf", [block])

    serialized = document.to_dict()
    serialized["blocks"][0]["metadata"]["source"] = "changed"

    assert document.blocks[0].metadata["source"] == "native"
