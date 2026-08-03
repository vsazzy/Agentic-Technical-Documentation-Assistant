import json
from pathlib import Path

import pytest

from document_models import ContentBlock, ContentType, NormalizedDocument
from vision_enrichment import (
    PageProfile,
    PdfPageRenderer,
    VisionClient,
    VisionDescription,
    VisionEnricher,
    should_enrich,
)


FIXTURE = Path(__file__).parent / "fixtures" / "vision_response.json"


class FakeRenderer:
    def __init__(self, pages: int = 3):
        self.pages = pages
        self.rendered_pages = []

    def page_count(self, pdf_path: Path) -> int:
        return self.pages

    def render_page(self, pdf_path: Path, page_number: int) -> bytes:
        self.rendered_pages.append((pdf_path, page_number))
        return f"png-page-{page_number}".encode()


class FakeVisionClient:
    def __init__(self, *, fail_pages=()):
        self.fail_pages = set(fail_pages)
        self.calls = []

    def describe(self, page_png: bytes, nearby_text: str) -> VisionDescription:
        page_number = int(page_png.decode().rsplit("-", 1)[1])
        self.calls.append((page_number, nearby_text))
        if page_number in self.fail_pages:
            raise RuntimeError("model unavailable")
        return VisionDescription(
            summary=f"Connection diagram on page {page_number}",
            visible_text=("WAN", "LAN"),
            relationships=("WAN connects to the uplink port",),
            steps=("Connect WAN", "Verify link light"),
            confidence=0.91,
        )


@pytest.fixture
def document():
    return NormalizedDocument(
        document_id="sha256:abc",
        filename="guide.pdf",
        blocks=(
            ContentBlock(
                "text-1",
                ContentType.TEXT,
                "Dense native text. " * 20,
                1,
                1,
                section_path=("Install",),
            ),
            ContentBlock(
                "image-2",
                ContentType.IMAGE,
                "Figure 1: Port layout",
                2,
                2,
                section_path=("Install",),
            ),
            ContentBlock(
                "text-3",
                ContentType.TEXT,
                "Connect the cables as shown.",
                3,
                3,
                section_path=("Install", "Cabling"),
            ),
        ),
    )


@pytest.mark.parametrize(
    "page",
    [
        PageProfile(page_number=1, native_text_chars=30, has_image=False),
        PageProfile(page_number=2, native_text_chars=500, has_image=True),
    ],
)
def test_should_enrich_low_text_or_image_pages(page):
    assert should_enrich(page)


def test_should_skip_dense_text_only_pages():
    assert not should_enrich(
        PageProfile(page_number=1, native_text_chars=500, has_image=False)
    )


def test_pdf_page_renderer_reports_count_and_emits_png(tmp_path):
    import fitz

    pdf_path = tmp_path / "guide.pdf"
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Local vision fixture")
        pdf.save(pdf_path)

    renderer = PdfPageRenderer(dpi=72)

    assert renderer.page_count(pdf_path) == 1
    assert renderer.render_page(pdf_path, 1).startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    "base_url",
    [
        "https://example.com",
        "http://192.168.1.20:11434",
        "http://127.0.0.1.example.com:11434",
    ],
)
def test_remote_ollama_url_is_rejected(base_url):
    with pytest.raises(ValueError, match="local"):
        VisionClient(base_url, "model")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.0.0.1:11434",
        "http://localhost:11434/",
        "http://[::1]:11434",
    ],
)
def test_loopback_ollama_url_is_accepted(base_url):
    VisionClient(base_url, "model")


def test_client_posts_image_and_context_and_validates_structured_json():
    response = json.loads(FIXTURE.read_text(encoding="utf-8"))
    requests = []

    def transport(url, payload, timeout):
        requests.append((url, payload, timeout))
        return response

    description = VisionClient(
        "http://127.0.0.1:11434",
        "qwen2.5vl:7b",
        transport=transport,
        timeout=7.5,
    ).describe(b"png-bytes", "Figure 1: Ports")

    assert description == VisionDescription(
        summary="A front-panel connection diagram.",
        visible_text=("WAN", "LAN1", "LAN2"),
        relationships=("WAN connects the appliance to the uplink.",),
        steps=("Connect WAN.", "Connect a workstation to LAN1."),
        confidence=0.94,
    )
    assert requests == [
        (
            "http://127.0.0.1:11434/api/chat",
            {
                "model": "qwen2.5vl:7b",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Describe this PDF page for retrieval. Use the nearby native "
                            "text only as context; do not replace or transcribe it wholesale.\n\n"
                            "Nearby native text:\nFigure 1: Ports"
                        ),
                        "images": ["cG5nLWJ5dGVz"],
                    }
                ],
                "stream": False,
                "format": VisionClient.response_schema(),
            },
            7.5,
        )
    ]


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps(
            {
                "summary": "diagram",
                "visible_text": ["WAN"],
                "relationships": [],
                "steps": [],
                "confidence": 1.5,
            }
        ),
        json.dumps(
            {
                "summary": "diagram",
                "visible_text": "WAN",
                "relationships": [],
                "steps": [],
                "confidence": 0.8,
            }
        ),
    ],
)
def test_client_rejects_invalid_structured_json(content):
    def transport(url, payload, timeout):
        return {"message": {"role": "assistant", "content": content}}

    with pytest.raises(ValueError, match="vision response"):
        VisionClient("http://localhost:11434", "model", transport=transport).describe(
            b"png", "context"
        )


def test_enrichment_adds_page_grounded_figures_without_replacing_native_blocks(document):
    renderer = FakeRenderer()
    client = FakeVisionClient()

    enriched = VisionEnricher(client, renderer).enrich(document, Path("guide.pdf"))

    figures = [block for block in enriched.blocks if block.content_type is ContentType.FIGURE]
    assert enriched.blocks[: len(document.blocks)] == document.blocks
    assert [(figure.page_start, figure.page_end) for figure in figures] == [(2, 2), (3, 3)]
    assert all(figure.extraction_method == "ollama_vision" for figure in figures)
    assert figures[0].metadata["source_block_ids"] == ("image-2",)
    assert figures[1].section_path == ("Install", "Cabling")
    assert renderer.rendered_pages == [(Path("guide.pdf"), 2), (Path("guide.pdf"), 3)]


def test_enrichment_records_page_warning_and_keeps_other_pages(document):
    renderer = FakeRenderer()
    client = FakeVisionClient(fail_pages={2})

    enriched = VisionEnricher(client, renderer).enrich(document, Path("guide.pdf"))

    figures = [block for block in enriched.blocks if block.content_type is ContentType.FIGURE]
    assert [figure.page_start for figure in figures] == [3]
    assert enriched.warnings == (
        "vision enrichment failed on page 2: model unavailable",
    )
    assert enriched.blocks[: len(document.blocks)] == document.blocks


def test_enrichment_appends_to_existing_warnings(document):
    document_with_warning = NormalizedDocument(
        document.document_id,
        document.filename,
        document.blocks,
        warnings=("OCR warning",),
    )

    enriched = VisionEnricher(
        FakeVisionClient(fail_pages={2}), FakeRenderer()
    ).enrich(document_with_warning, Path("guide.pdf"))

    assert enriched.warnings == (
        "OCR warning",
        "vision enrichment failed on page 2: model unavailable",
    )


def test_enrichment_captures_render_failures_as_warnings(document):
    class BrokenRenderer(FakeRenderer):
        def render_page(self, pdf_path: Path, page_number: int) -> bytes:
            if page_number == 2:
                raise ValueError("page cannot be rendered")
            return super().render_page(pdf_path, page_number)

    enriched = VisionEnricher(FakeVisionClient(), BrokenRenderer()).enrich(
        document, Path("guide.pdf")
    )

    assert [b.page_start for b in enriched.blocks if b.content_type is ContentType.FIGURE] == [3]
    assert enriched.warnings == (
        "vision enrichment failed on page 2: page cannot be rendered",
    )
