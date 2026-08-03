import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from document_models import ContentType
from pdf_extractor import PdfExtractionError, PdfExtractor


FIXTURE = Path(__file__).parent / "fixtures" / "docling_document.json"


class FakeBackend:
    def __init__(self, document):
        self.document = document

    def convert(self, path: Path):
        return self.document


@pytest.fixture
def docling_document():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def fake_backend(docling_document):
    return FakeBackend(docling_document)


def test_extract_preserves_reading_order_sections_and_structured_content(fake_backend, tmp_path):
    result = PdfExtractor(fake_backend).extract(tmp_path / "guide.pdf", "sha256:abc")

    assert result.filename == "guide.pdf"
    assert [block.text.splitlines()[0] for block in result.blocks] == [
        "Installation",
        "Connect the appliance before configuration.",
        "| Port | Purpose |",
        "Verify the status LED.",
        "OCR serial: A1-2048",
        "Figure 1: Front-panel connections",
    ]
    assert [block.section_path for block in result.blocks] == [
        ("Installation",),
        ("Installation",),
        ("Installation",),
        ("Installation",),
        ("Installation",),
        ("Installation",),
    ]
    assert [block.content_type for block in result.blocks] == [
        ContentType.TEXT,
        ContentType.TEXT,
        ContentType.TABLE,
        ContentType.TEXT,
        ContentType.TEXT,
        ContentType.IMAGE,
    ]


def test_extract_preserves_table_page_range_markdown_and_provenance(fake_backend, tmp_path):
    result = PdfExtractor(fake_backend).extract(tmp_path / "guide.pdf", "sha256:abc")

    table = next(block for block in result.blocks if block.content_type is ContentType.TABLE)
    assert table.section_path == ("Installation",)
    assert (table.page_start, table.page_end) == (2, 3)
    assert "| Port | Purpose |" in table.text
    assert "| ETH1 | Data |" in table.text
    assert table.metadata["label"] == "table"
    assert table.metadata["provenance"] == (
        {
            "page": 2,
            "bbox": {"left": 55.0, "top": 180.0, "right": 500.0, "bottom": 720.0, "origin": "TOPLEFT"},
        },
        {
            "page": 3,
            "bbox": {"left": 55.0, "top": 70.0, "right": 500.0, "bottom": 350.0, "origin": "TOPLEFT"},
        },
    )


def test_extract_uses_one_based_pages_and_records_extraction_method(fake_backend, tmp_path):
    result = PdfExtractor(fake_backend).extract(tmp_path / "guide.pdf", "sha256:abc")

    assert result.stats.pages == 4
    assert all(block.page_start >= 1 for block in result.blocks)
    assert all(block.extraction_method == "docling" for block in result.blocks)
    ocr = next(block for block in result.blocks if block.metadata["label"] == "handwritten_text")
    assert (ocr.page_start, ocr.page_end) == (4, 4)


def test_extract_generates_stable_ids_from_document_and_block_location(fake_backend, tmp_path):
    extractor = PdfExtractor(fake_backend)
    path = tmp_path / "guide.pdf"

    first = extractor.extract(path, "sha256:abc")
    repeated = extractor.extract(path, "sha256:abc")
    other_document = extractor.extract(path, "sha256:def")

    assert [block.block_id for block in first.blocks] == [block.block_id for block in repeated.blocks]
    assert len({block.block_id for block in first.blocks}) == len(first.blocks)
    assert [block.block_id for block in first.blocks] != [block.block_id for block in other_document.blocks]


def test_extract_wraps_conversion_and_normalization_failures(tmp_path, docling_document):
    class BrokenBackend:
        def convert(self, path: Path):
            raise RuntimeError("model unavailable")

    with pytest.raises(PdfExtractionError, match="model unavailable") as conversion:
        PdfExtractor(BrokenBackend()).extract(tmp_path / "guide.pdf", "sha256:abc")
    assert conversion.value.stage == "convert"

    docling_document["texts"][0]["prov"][0]["page_no"] = 0
    with pytest.raises(PdfExtractionError, match="one-based") as normalization:
        PdfExtractor(FakeBackend(docling_document)).extract(tmp_path / "guide.pdf", "sha256:abc")
    assert normalization.value.stage == "normalize"


def test_extract_rejects_documents_without_normalized_blocks(tmp_path):
    empty = {"body": {"children": []}, "texts": [], "tables": [], "pictures": []}

    with pytest.raises(PdfExtractionError, match="no supported content") as error:
        PdfExtractor(FakeBackend(empty)).extract(tmp_path / "empty.pdf", "sha256:empty")

    assert error.value.stage == "empty_document"


def test_extract_resolves_relative_paths_before_store_validation(fake_backend, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def require_absolute_path(path: Path):
        if not path.is_absolute():
            raise ValueError("validation requires an absolute path")
        return SimpleNamespace(filename="guide.pdf")

    result = PdfExtractor(fake_backend, validate_pdf=require_absolute_path).extract(
        Path("guide.pdf"),
        "sha256:abc",
    )

    assert result.filename == "guide.pdf"


def test_importing_adapter_does_not_import_docling():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pdf_extractor; "
            "assert not any(n == 'docling' or n.startswith('docling.') for n in sys.modules)",
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
