"""Normalized document contracts shared by extraction and indexing workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from types import MappingProxyType


class ContentType(str, Enum):
    """The supported normalized content block types."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FIGURE = "figure"


def _require_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _validate_page_range(page_start: int, page_end: int) -> None:
    if not isinstance(page_start, int) or isinstance(page_start, bool) or page_start < 1:
        raise ValueError("page_start must be a one-based page number")
    if not isinstance(page_end, int) or isinstance(page_end, bool) or page_end < page_start:
        raise ValueError("page_end must be greater than or equal to page_start")


def _normalize_content_type(content_type: ContentType | str) -> ContentType:
    try:
        return ContentType(content_type)
    except (TypeError, ValueError) as error:
        raise ValueError(f"content_type must be one of: {', '.join(t.value for t in ContentType)}") from error


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_metadata(item) for item in value)
    return value


def _serialize_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _serialize_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serialize_metadata(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_serialize_metadata(item) for item in value)
    return value


@dataclass(frozen=True)
class ContentBlock:
    block_id: str
    content_type: ContentType
    text: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extraction_method: str = "docling"

    def __post_init__(self) -> None:
        _require_non_empty(self.block_id, "block_id")
        _require_non_empty(self.text, "text")
        _require_non_empty(self.extraction_method, "extraction_method")
        _validate_page_range(self.page_start, self.page_end)
        object.__setattr__(self, "content_type", _normalize_content_type(self.content_type))
        object.__setattr__(self, "section_path", tuple(self.section_path))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "content_type": self.content_type.value,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_path": list(self.section_path),
            "metadata": _serialize_metadata(self.metadata),
            "extraction_method": self.extraction_method,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ExtractionStats:
    pages: int = 0
    blocks: int = 0
    text_blocks: int = 0
    tables: int = 0
    images: int = 0


@dataclass(frozen=True)
class NormalizedDocument:
    document_id: str
    filename: str
    blocks: tuple[ContentBlock, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.document_id, "document_id")
        _require_non_empty(self.filename, "filename")
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def stats(self) -> ExtractionStats:
        return ExtractionStats(
            pages=max((block.page_end for block in self.blocks), default=0),
            blocks=len(self.blocks),
            text_blocks=sum(block.content_type is ContentType.TEXT for block in self.blocks),
            tables=sum(block.content_type is ContentType.TABLE for block in self.blocks),
            images=sum(
                block.content_type in {ContentType.IMAGE, ContentType.FIGURE}
                for block in self.blocks
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "blocks": [block.to_dict() for block in self.blocks],
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class IndexChunk:
    chunk_id: str
    document_id: str
    filename: str
    content_type: ContentType
    text: str
    page_start: int
    page_end: int
    section_path: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    extraction_method: str = "docling"

    def __post_init__(self) -> None:
        _require_non_empty(self.chunk_id, "chunk_id")
        _require_non_empty(self.document_id, "document_id")
        _require_non_empty(self.filename, "filename")
        _require_non_empty(self.text, "text")
        _require_non_empty(self.extraction_method, "extraction_method")
        _validate_page_range(self.page_start, self.page_end)
        object.__setattr__(self, "content_type", _normalize_content_type(self.content_type))
        object.__setattr__(self, "section_path", tuple(self.section_path))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "content_type": self.content_type.value,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_path": list(self.section_path),
            "metadata": _serialize_metadata(self.metadata),
            "extraction_method": self.extraction_method,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
