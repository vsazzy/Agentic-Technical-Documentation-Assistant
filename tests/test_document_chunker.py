from document_models import ContentBlock, ContentType, NormalizedDocument
from document_chunker import DocumentChunker


def _document(*blocks, warnings=()):
    return NormalizedDocument(
        document_id="sha256:chunking-fixture",
        filename="installation-guide.pdf",
        blocks=blocks,
        warnings=warnings,
    )


def test_table_is_not_split_when_under_limit():
    document = _document(
        ContentBlock(
            "table-1",
            ContentType.TABLE,
            "| Port | Purpose |\n| --- | --- |\n| WAN | Uplink |",
            2,
            2,
            section_path=("Installation", "Ports"),
        )
    )

    chunks = DocumentChunker(max_chars=1_000, overlap_chars=100).build(document)

    table_chunks = [chunk for chunk in chunks if chunk.content_type is ContentType.TABLE]
    assert len(table_chunks) == 1
    assert table_chunks[0].text.startswith("[TABLE] Section: Installation > Ports\n")
    assert "| WAN | Uplink |" in table_chunks[0].text
    assert table_chunks[0].metadata["block_ids"] == ("table-1",)


def test_adjacent_walkthrough_steps_stay_in_one_text_chunk():
    document = _document(
        ContentBlock("step-1", ContentType.TEXT, "1. Connect the WAN cable.", 3, 3, ("Setup",)),
        ContentBlock("step-2", ContentType.TEXT, "2. Power on the appliance.", 3, 3, ("Setup",)),
        ContentBlock("step-3", ContentType.TEXT, "3. Verify the status LED.", 3, 3, ("Setup",)),
    )

    chunks = DocumentChunker(max_chars=400, overlap_chars=50).build(document)

    assert len(chunks) == 1
    assert "1. Connect the WAN cable." in chunks[0].text
    assert "2. Power on the appliance." in chunks[0].text
    assert "3. Verify the status LED." in chunks[0].text
    assert chunks[0].metadata["block_ids"] == ("step-1", "step-2", "step-3")


def test_figure_chunk_keeps_source_caption_with_vision_description():
    document = _document(
        ContentBlock(
            "image-1",
            ContentType.IMAGE,
            "Figure 1: Front-panel connections",
            4,
            4,
            ("Cabling",),
        ),
        ContentBlock(
            "vision-1",
            ContentType.FIGURE,
            "Summary: A front-panel diagram.\nRelationships: WAN connects to the uplink.",
            4,
            4,
            ("Cabling",),
            metadata={"source_block_ids": ("image-1",)},
            extraction_method="ollama_vision",
        ),
    )

    chunks = DocumentChunker(max_chars=500, overlap_chars=50).build(document)

    figure = next(chunk for chunk in chunks if chunk.content_type is ContentType.FIGURE)
    assert figure.text.startswith("[FIGURE] Section: Cabling\nCaption: Figure 1: Front-panel connections")
    assert "Summary: A front-panel diagram." in figure.text
    assert figure.metadata["block_ids"] == ("vision-1",)
    assert figure.metadata["extraction_methods"] == ("ollama_vision",)


def test_oversized_block_splits_with_overlap():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    document = _document(ContentBlock("long-1", ContentType.TEXT, text, 1, 1, ("Overview",)))

    chunks = DocumentChunker(max_chars=54, overlap_chars=12).build(document)

    assert len(chunks) > 1
    bodies = [chunk.text.split("\n", 1)[1] for chunk in chunks]
    assert all(len(chunk.text) <= 54 for chunk in chunks)
    assert any(later[:12] == earlier[-12:] for earlier, later in zip(bodies, bodies[1:]))
    assert all(chunk.metadata["block_ids"] == ("long-1",) for chunk in chunks)


def test_chunk_ids_are_deterministic_and_metadata_preserves_provenance_and_warnings():
    document = _document(
        ContentBlock(
            "text-1",
            ContentType.TEXT,
            "Connect the appliance before configuration.",
            1,
            2,
            ("Installation",),
            extraction_method="docling",
        ),
        warnings=("vision enrichment failed on page 2",),
    )
    chunker = DocumentChunker(max_chars=1_000, overlap_chars=100)

    first = chunker.build(document)
    second = chunker.build(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert first[0].metadata == {
        "page_start": 1,
        "page_end": 2,
        "content_type": "text",
        "section_path": ("Installation",),
        "extraction_methods": ("docling",),
        "block_ids": ("text-1",),
        "warnings": ("vision enrichment failed on page 2",),
    }
