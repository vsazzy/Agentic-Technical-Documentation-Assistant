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
    assert figure.metadata["block_ids"] == ("image-1", "vision-1")
    assert figure.metadata["extraction_methods"] == ("docling", "ollama_vision")


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


def test_deep_section_path_uses_a_bounded_display_prefix_and_keeps_full_metadata():
    section_path = ("Very long heading " * 4, "Another long heading " * 4)
    document = _document(
        ContentBlock("text-1", ContentType.TEXT, "Short body.", 1, 1, section_path)
    )

    chunk = DocumentChunker(max_chars=32, overlap_chars=8).build(document)[0]

    assert len(chunk.text) <= 32
    assert chunk.text.startswith("[TEXT]")
    assert chunk.metadata["section_path"] == section_path


def test_extractor_list_markers_are_rendered_once_and_remain_grouped():
    document = _document(
        ContentBlock(
            "step-1",
            ContentType.TEXT,
            "Connect the WAN cable.",
            1,
            1,
            ("Setup",),
            metadata={"marker": "1.", "enumerated": True},
        ),
        ContentBlock(
            "step-2",
            ContentType.TEXT,
            "2. Power on the appliance.",
            1,
            1,
            ("Setup",),
            metadata={"marker": "2.", "enumerated": True},
        ),
        ContentBlock(
            "step-3",
            ContentType.TEXT,
            "Verify the status LED.",
            1,
            1,
            ("Setup",),
            metadata={"marker": "3.", "enumerated": True},
        ),
    )

    chunk = DocumentChunker(max_chars=400, overlap_chars=50).build(document)[0]

    assert "1. Connect the WAN cable." in chunk.text
    assert chunk.text.count("2. Power on the appliance.") == 1
    assert "3. Verify the status LED." in chunk.text


def test_figure_caption_filters_page_noise_and_records_all_incorporated_provenance():
    document = _document(
        ContentBlock("prose", ContentType.TEXT, "Unrelated page prose.", 2, 2, ("Cabling",)),
        ContentBlock("table", ContentType.TABLE, "| Noise |", 2, 2, ("Cabling",)),
        ContentBlock("other-image", ContentType.IMAGE, "Figure 9: Other section", 2, 2, ("Safety",)),
        ContentBlock("caption", ContentType.IMAGE, "Figure 1: WAN port", 2, 2, ("Cabling",)),
        ContentBlock(
            "vision",
            ContentType.FIGURE,
            "Summary: The WAN port is on the left.",
            2,
            2,
            ("Cabling",),
            metadata={"source_block_ids": ("prose", "table", "other-image", "caption")},
            extraction_method="ollama_vision",
        ),
    )

    figure = next(
        chunk
        for chunk in DocumentChunker(max_chars=500, overlap_chars=30).build(document)
        if chunk.content_type is ContentType.FIGURE
    )

    assert "Caption: Figure 1: WAN port" in figure.text
    assert "Unrelated page prose" not in figure.text
    assert "| Noise |" not in figure.text
    assert "Figure 9: Other section" not in figure.text
    assert figure.metadata["block_ids"] == ("caption", "vision")
    assert figure.metadata["extraction_methods"] == ("docling", "ollama_vision")


def test_figure_uses_nearby_same_section_caption_when_references_are_stale():
    document = _document(
        ContentBlock("nearby", ContentType.IMAGE, "Figure 2: Status lights", 3, 3, ("Status",)),
        ContentBlock("other", ContentType.IMAGE, "Figure 8: Safety label", 3, 3, ("Safety",)),
        ContentBlock(
            "vision",
            ContentType.FIGURE,
            "Summary: LEDs indicate readiness.",
            3,
            3,
            ("Status",),
            metadata={"source_block_ids": ("missing-id",)},
            extraction_method="ollama_vision",
        ),
    )

    figure = next(
        chunk
        for chunk in DocumentChunker(max_chars=500, overlap_chars=30).build(document)
        if chunk.content_type is ContentType.FIGURE
    )

    assert "Caption: Figure 2: Status lights" in figure.text
    assert "Figure 8: Safety label" not in figure.text
    assert figure.metadata["block_ids"] == ("nearby", "vision")


def test_long_figure_caption_splits_bounded_chunks_with_complete_provenance():
    caption_text = "Figure 3: " + "front-panel connector detail " * 8
    document = _document(
        ContentBlock("caption", ContentType.IMAGE, caption_text, 4, 4, ("Ports",)),
        ContentBlock(
            "vision",
            ContentType.FIGURE,
            "Summary: The WAN connector is highlighted.",
            4,
            4,
            ("Ports",),
            metadata={"source_block_ids": ("caption",)},
            extraction_method="ollama_vision",
        ),
    )

    all_chunks = DocumentChunker(max_chars=56, overlap_chars=8).build(document)
    chunks = [
        chunk
        for chunk in all_chunks
        if chunk.content_type is ContentType.FIGURE
    ]

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 56 for chunk in all_chunks)
    rendered = "\n".join(chunk.text for chunk in chunks)
    assert "Description:" in rendered
    assert "Summary:" in rendered
    assert all(chunk.metadata["block_ids"] == ("caption", "vision") for chunk in chunks)


def test_retrieval_labels_distinguish_text_table_ocr_and_figure_without_image_label():
    document = _document(
        ContentBlock("text", ContentType.TEXT, "Plain prose.", 1, 1, ("Text",)),
        ContentBlock(
            "ocr", ContentType.TEXT, "Serial A1-2048", 2, 2, ("OCR",), metadata={"label": "handwritten_text"}
        ),
        ContentBlock("table", ContentType.TABLE, "| A |", 3, 3, ("Table",)),
        ContentBlock("image", ContentType.IMAGE, "Figure 1: Native image", 4, 4, ("Image",)),
        ContentBlock("figure", ContentType.FIGURE, "Summary: Vision image", 5, 5, ("Figure",)),
    )

    labels = [chunk.text.split(" ", 1)[0] for chunk in DocumentChunker(500, 30).build(document)]

    assert labels == ["[TEXT]", "[OCR]", "[TABLE]", "[FIGURE]", "[FIGURE]"]
    assert "[IMAGE]" not in labels


def test_every_overflow_boundary_in_a_compatible_run_has_overlap_without_crossing_sections():
    document = _document(
        ContentBlock("step-1", ContentType.TEXT, "first walkthrough step", 1, 1, ("Setup",)),
        ContentBlock("step-2", ContentType.TEXT, "second walkthrough step", 1, 1, ("Setup",)),
        ContentBlock("step-3", ContentType.TEXT, "third walkthrough step", 1, 1, ("Setup",)),
        ContentBlock("other", ContentType.TEXT, "different section starts cleanly", 1, 1, ("Safety",)),
    )

    chunks = DocumentChunker(max_chars=52, overlap_chars=8).build(document)
    setup_chunks = [chunk for chunk in chunks if chunk.section_path == ("Setup",)]
    bodies = [chunk.text.split("\n", 1)[1] for chunk in setup_chunks]

    assert len(setup_chunks) >= 3
    assert all(later[:8] == earlier[-8:] for earlier, later in zip(bodies, bodies[1:]))
    assert chunks[-1].section_path == ("Safety",)
    assert not chunks[-1].text.split("\n", 1)[1].startswith(bodies[-1][-8:])
