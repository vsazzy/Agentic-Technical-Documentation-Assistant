"""Local Docling adapter for normalized, provenance-preserving PDF extraction."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from document_models import ContentBlock, ContentType, NormalizedDocument


class ExtractionBackend(Protocol):
    """The conversion boundary used by :class:`PdfExtractor`."""

    def convert(self, path: Path) -> object:
        """Convert one PDF into a Docling-shaped document."""


class PdfExtractionError(RuntimeError):
    """A per-document extraction failure with a machine-readable stage."""

    def __init__(
        self,
        stage: Literal["convert", "normalize", "empty_document"],
        message: str,
    ) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"{stage}: {message}")


class DoclingBackend:
    """Lazily configured, offline Docling PDF conversion backend."""

    def __init__(self, converter: object | None = None) -> None:
        self._converter = converter

    @staticmethod
    def _build_converter() -> object:
        # Keep the heavyweight optional dependency behind the backend boundary.
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            OcrAutoOptions,
            PdfPipelineOptions,
            TableFormerMode,
            TableStructureOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=OcrAutoOptions(),
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,
                mode=TableFormerMode.ACCURATE,
            ),
            enable_remote_services=False,
            allow_external_plugins=False,
        )
        return DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )

    def convert(self, path: Path) -> object:
        """Return the converted Docling document or raise on conversion failure."""
        if self._converter is None:
            self._converter = self._build_converter()

        result = self._converter.convert(Path(path))  # type: ignore[attr-defined]
        status = _enum_value(getattr(result, "status", "success"))
        if status not in {"success", "partial_success"}:
            errors = getattr(result, "errors", ())
            detail = "; ".join(_conversion_error_message(error) for error in errors)
            raise RuntimeError(detail or f"Docling conversion ended with status {status}")
        document = getattr(result, "document", None)
        if document is None:
            raise RuntimeError("Docling conversion did not return a document")
        return document


class PdfExtractor:
    """Normalize Docling reading-order items into immutable document blocks."""

    def __init__(
        self,
        backend: ExtractionBackend,
        *,
        validate_pdf: Callable[[Path], object] | None = None,
    ) -> None:
        self._backend = backend
        self._validate_pdf = validate_pdf

    @classmethod
    def local(cls) -> "PdfExtractor":
        """Build the validated, fully local extraction pipeline."""
        from document_store import DocumentStore

        store = DocumentStore()
        return cls(DoclingBackend(), validate_pdf=store.validate_pdf)

    def extract(self, path: Path, document_id: str) -> NormalizedDocument:
        """Convert and normalize one PDF without hiding per-document failures."""
        pdf_path = Path(path).resolve()
        filename = pdf_path.name
        try:
            if self._validate_pdf is not None:
                record = self._validate_pdf(pdf_path)
                filename = str(getattr(record, "filename", filename))
            converted = self._backend.convert(pdf_path)
        except Exception as error:
            raise PdfExtractionError("convert", str(error) or type(error).__name__) from error

        try:
            document = _unwrap_document(converted)
            blocks = tuple(_normalize_blocks(document, document_id))
            normalized = NormalizedDocument(
                document_id=document_id,
                filename=filename,
                blocks=blocks,
            )
        except PdfExtractionError:
            raise
        except Exception as error:
            raise PdfExtractionError("normalize", str(error) or type(error).__name__) from error

        if not normalized.blocks:
            raise PdfExtractionError("empty_document", "no supported content was extracted")
        return normalized


def _normalize_blocks(document: object, document_id: str) -> Iterable[ContentBlock]:
    section_stack: list[tuple[int, str]] = []
    caption_refs = _picture_caption_refs(document)
    ordinal = 0

    for item in _iterate_items(document):
        label = _label(item)
        item_ref = _self_ref(item)
        if label == "caption" and item_ref in caption_refs:
            continue

        content_type, text = _item_content(document, item, label)
        if content_type is None or not text.strip():
            continue

        if label in {"section_header", "title"}:
            level = 0 if label == "title" else _heading_level(item)
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, text.strip()))

        provenance = _provenance(item)
        if not provenance:
            raise ValueError(f"{label or 'document item'} has no page provenance")
        pages = [entry["page"] for entry in provenance]
        if any(not isinstance(page, int) or isinstance(page, bool) or page < 1 for page in pages):
            raise ValueError("Docling page provenance must use one-based page numbers")
        page_start, page_end = min(pages), max(pages)
        ordinal += 1

        metadata: dict[str, Any] = {
            "label": label,
            "provenance": provenance,
        }
        if label == "list_item":
            metadata["marker"] = str(_get(item, "marker", "-"))
            metadata["enumerated"] = bool(_get(item, "enumerated", False))
        if label in {"section_header", "title"}:
            metadata["heading_level"] = 0 if label == "title" else _heading_level(item)

        yield ContentBlock(
            block_id=_block_id(
                document_id=document_id,
                content_type=content_type,
                page_start=page_start,
                page_end=page_end,
                ordinal=ordinal,
            ),
            content_type=content_type,
            text=text.strip(),
            page_start=page_start,
            page_end=page_end,
            section_path=tuple(title for _, title in section_stack),
            metadata=metadata,
            extraction_method="docling",
        )


def _unwrap_document(converted: object) -> object:
    if isinstance(converted, Mapping):
        if "body" in converted:
            return converted
        nested = converted.get("document")
        if nested is not None:
            return nested
    nested = getattr(converted, "document", None)
    return nested if nested is not None else converted


def _iterate_items(document: object) -> Iterable[object]:
    if not isinstance(document, Mapping):
        iterate_items = getattr(document, "iterate_items", None)
        if not callable(iterate_items):
            raise TypeError("converted document does not expose Docling reading order")
        for item, _level in iterate_items(with_groups=False, traverse_pictures=False):
            yield item
        return

    body = document.get("body")
    if not isinstance(body, Mapping):
        raise TypeError("converted document has no Docling body")

    def walk(references: object) -> Iterable[object]:
        if not _is_sequence(references):
            return
        for reference in references:
            item = _resolve_ref(document, reference)
            yield item
            yield from walk(_get(item, "children", ()))

    yield from walk(body.get("children", ()))


def _item_content(document: object, item: object, label: str) -> tuple[ContentType | None, str]:
    if label in {"table", "document_index"}:
        if isinstance(item, Mapping):
            return ContentType.TABLE, _mapping_table_markdown(item)
        export = getattr(item, "export_to_markdown", None)
        if not callable(export):
            raise TypeError("Docling table item cannot export Markdown")
        return ContentType.TABLE, str(export(doc=document))

    if label in {"picture", "chart"}:
        caption = _picture_caption(document, item)
        return ContentType.IMAGE, caption or ("Chart" if label == "chart" else "Image")

    text = _get(item, "text", None)
    if isinstance(text, str):
        return ContentType.TEXT, text
    return None, ""


def _mapping_table_markdown(item: Mapping[str, Any]) -> str:
    data = item.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Docling table has no structured data")
    row_count = _positive_int(data.get("num_rows"), "table row count")
    column_count = _positive_int(data.get("num_cols"), "table column count")
    grid = [["" for _ in range(column_count)] for _ in range(row_count)]
    cells = data.get("table_cells", ())
    if not _is_sequence(cells):
        raise ValueError("Docling table cells are malformed")

    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ValueError("Docling table cell is malformed")
        row_start = int(cell.get("start_row_offset_idx", 0))
        row_end = int(cell.get("end_row_offset_idx", row_start + 1))
        column_start = int(cell.get("start_col_offset_idx", 0))
        column_end = int(cell.get("end_col_offset_idx", column_start + 1))
        value = _markdown_cell(str(cell.get("text", "")))
        for row in range(max(row_start, 0), min(row_end, row_count)):
            for column in range(max(column_start, 0), min(column_end, column_count)):
                grid[row][column] = value

    lines = [_markdown_row(grid[0])]
    lines.append(_markdown_row(["---"] * column_count))
    lines.extend(_markdown_row(row) for row in grid[1:])
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _markdown_row(values: Sequence[str]) -> str:
    return "| " + " | ".join(values) + " |"


def _picture_caption(document: object, item: object) -> str:
    if not isinstance(item, Mapping):
        caption_text = getattr(item, "caption_text", None)
        if callable(caption_text):
            caption = str(caption_text(document)).strip()
            if caption:
                return caption
    else:
        captions = item.get("captions", ())
        if _is_sequence(captions):
            caption_parts = [str(_get(_resolve_ref(document, ref), "text", "")).strip() for ref in captions]
            caption = " ".join(part for part in caption_parts if part)
            if caption:
                return caption

    meta = _get(item, "meta", None)
    description = _get(meta, "description", None)
    return str(_get(description, "text", "")).strip()


def _picture_caption_refs(document: object) -> set[str]:
    references: set[str] = set()
    if isinstance(document, Mapping):
        pictures = document.get("pictures", ())
    else:
        pictures = getattr(document, "pictures", ())
    if not _is_sequence(pictures):
        return references
    for picture in pictures:
        captions = _get(picture, "captions", ())
        if not _is_sequence(captions):
            continue
        for caption in captions:
            ref = _ref_value(caption)
            if ref:
                references.add(ref)
    return references


def _provenance(item: object) -> list[dict[str, Any]]:
    raw_provenance = _get(item, "prov", ())
    if not _is_sequence(raw_provenance):
        raise ValueError("Docling provenance is malformed")
    normalized: list[dict[str, Any]] = []
    for entry in raw_provenance:
        page = _get(entry, "page_no", None)
        bbox = _get(entry, "bbox", None)
        normalized_bbox = {
            "left": _get(bbox, "l", None),
            "top": _get(bbox, "t", None),
            "right": _get(bbox, "r", None),
            "bottom": _get(bbox, "b", None),
            "origin": _enum_value(_get(bbox, "coord_origin", "TOPLEFT")),
        }
        normalized.append({"page": page, "bbox": normalized_bbox})
    return normalized


def _resolve_ref(document: object, reference: object) -> object:
    if not isinstance(document, Mapping):
        resolve = getattr(reference, "resolve", None)
        if callable(resolve):
            return resolve(document)
        raise TypeError("Docling item reference cannot be resolved")
    ref = _ref_value(reference)
    if not ref:
        if isinstance(reference, Mapping) and "label" in reference:
            return reference
        raise ValueError("Docling item reference is malformed")
    parts = ref.removeprefix("#/").split("/")
    if len(parts) != 2:
        raise ValueError(f"unsupported Docling reference: {ref}")
    collection, raw_index = parts
    values = document.get(collection)
    if not _is_sequence(values):
        raise ValueError(f"Docling reference collection is missing: {collection}")
    return values[int(raw_index)]


def _ref_value(reference: object) -> str:
    value = _get(reference, "cref", None)
    if value is None:
        value = _get(reference, "$ref", "")
    return str(value)


def _self_ref(item: object) -> str:
    return str(_get(item, "self_ref", ""))


def _label(item: object) -> str:
    return _enum_value(_get(item, "label", ""))


def _heading_level(item: object) -> int:
    value = _get(item, "level", 1)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("Docling heading level must be a positive integer")
    return value


def _block_id(
    *,
    document_id: str,
    content_type: ContentType,
    page_start: int,
    page_end: int,
    ordinal: int,
) -> str:
    identity = f"{document_id}\0{content_type.value}\0{page_start}\0{page_end}\0{ordinal}"
    return f"block:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _get(value: object, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _conversion_error_message(error: object) -> str:
    for field in ("error_message", "message"):
        value = getattr(error, field, None)
        if value:
            return str(value)
    return str(error)
