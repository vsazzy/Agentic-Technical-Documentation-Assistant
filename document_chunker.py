"""Deterministic, structure-aware chunks for normalized PDF documents."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from document_models import ContentBlock, ContentType, IndexChunk, NormalizedDocument


class DocumentChunker:
    """Render normalized blocks without losing their semantic retrieval context."""

    _ATOMIC_TYPES = frozenset({ContentType.TABLE, ContentType.IMAGE, ContentType.FIGURE})
    _OCR_LABELS = frozenset({"ocr", "ocr_text", "handwritten_text"})
    _MIN_MAX_CHARS = len("[FIGURE]\n ")

    def __init__(self, max_chars: int, overlap_chars: int) -> None:
        if (
            not isinstance(max_chars, int)
            or isinstance(max_chars, bool)
            or max_chars < self._MIN_MAX_CHARS
        ):
            raise ValueError(f"max_chars must be an integer of at least {self._MIN_MAX_CHARS}")
        if (
            not isinstance(overlap_chars, int)
            or isinstance(overlap_chars, bool)
            or overlap_chars < 0
            or overlap_chars >= max_chars
        ):
            raise ValueError("overlap_chars must be non-negative and smaller than max_chars")
        self._max_chars = max_chars
        self._overlap_chars = overlap_chars

    def build(self, document: NormalizedDocument) -> list[IndexChunk]:
        """Build ordered, bounded chunks while preserving normalized provenance."""
        indexed_blocks = tuple(enumerate(document.blocks))
        block_by_id = {block.block_id: block for _, block in indexed_blocks}
        position_by_id = {block.block_id: position for position, block in indexed_blocks}
        chunks: list[IndexChunk] = []
        text_run: list[ContentBlock] = []
        split_ordinal = 0

        def emit_text_run() -> None:
            nonlocal split_ordinal
            if not text_run:
                return
            body, spans = self._body_from_blocks(text_run)
            emitted = self._make_chunks(
                document,
                ContentType.TEXT,
                self._retrieval_label(text_run[0]),
                text_run[0].section_path,
                body,
                spans,
                split_ordinal,
            )
            chunks.extend(emitted)
            split_ordinal += len(emitted)
            text_run.clear()

        for block in document.blocks:
            if block.content_type not in self._ATOMIC_TYPES:
                if text_run and not self._compatible(text_run[-1], block):
                    emit_text_run()
                text_run.append(block)
                continue

            emit_text_run()
            caption_blocks = (
                self._figure_captions(block, block_by_id, position_by_id, indexed_blocks)
                if block.content_type is ContentType.FIGURE
                else ()
            )
            body, spans = self._figure_body(block, caption_blocks)
            emitted = self._make_chunks(
                document,
                block.content_type,
                self._retrieval_label(block),
                block.section_path,
                body,
                spans,
                split_ordinal,
            )
            chunks.extend(emitted)
            split_ordinal += len(emitted)

        emit_text_run()
        return chunks

    def _compatible(self, previous: ContentBlock, current: ContentBlock) -> bool:
        return (
            previous.content_type is current.content_type
            and self._retrieval_label(previous) == self._retrieval_label(current)
            and previous.section_path == current.section_path
            and current.page_start <= previous.page_end + 1
        )

    @classmethod
    def _retrieval_label(cls, block: ContentBlock) -> str:
        if block.content_type is ContentType.TABLE:
            return "TABLE"
        if block.content_type in {ContentType.IMAGE, ContentType.FIGURE}:
            return "FIGURE"
        source_label = block.metadata.get("label")
        if isinstance(source_label, str) and source_label.casefold() in cls._OCR_LABELS:
            return "OCR"
        return "TEXT"

    @staticmethod
    def _block_text(block: ContentBlock) -> str:
        marker = block.metadata.get("marker")
        if not isinstance(marker, str) or not marker.strip():
            return block.text
        normalized_marker = marker.strip()
        leading_text = block.text.lstrip()
        if re.match(rf"^{re.escape(normalized_marker)}(?:\s|$)", leading_text):
            return block.text
        return f"{normalized_marker} {block.text}"

    def _figure_captions(
        self,
        figure: ContentBlock,
        block_by_id: dict[str, ContentBlock],
        position_by_id: dict[str, int],
        indexed_blocks: Sequence[tuple[int, ContentBlock]],
    ) -> tuple[ContentBlock, ...]:
        requested_ids = figure.metadata.get("source_block_ids", ())
        requested = [
            block_by_id[block_id]
            for block_id in requested_ids
            if isinstance(block_id, str)
            and block_id in block_by_id
            and self._is_caption_candidate(block_by_id[block_id], figure)
        ]
        if requested:
            return tuple(sorted(requested, key=lambda block: position_by_id[block.block_id]))

        nearby = [
            (position, block)
            for position, block in indexed_blocks
            if self._is_caption_candidate(block, figure)
        ]
        if not nearby:
            return ()
        figure_position = position_by_id[figure.block_id]
        _, closest = min(
            nearby,
            key=lambda pair: (
                self._page_distance(pair[1], figure),
                abs(pair[0] - figure_position),
                pair[0],
            ),
        )
        return (closest,)

    @staticmethod
    def _is_caption_candidate(candidate: ContentBlock, figure: ContentBlock) -> bool:
        if candidate.section_path != figure.section_path:
            return False
        is_native_visual = candidate.content_type is ContentType.IMAGE
        is_caption = candidate.metadata.get("label") == "caption"
        return (is_native_visual or is_caption) and DocumentChunker._page_distance(candidate, figure) <= 1

    @staticmethod
    def _page_distance(left: ContentBlock, right: ContentBlock) -> int:
        if left.page_end < right.page_start:
            return right.page_start - left.page_end
        if right.page_end < left.page_start:
            return left.page_start - right.page_end
        return 0

    def _make_chunks(
        self,
        document: NormalizedDocument,
        content_type: ContentType,
        retrieval_label: str,
        section_path: tuple[str, ...],
        body: str,
        spans: Sequence[tuple[int, int, ContentBlock]],
        first_split_ordinal: int,
    ) -> list[IndexChunk]:
        header = self._header(retrieval_label, section_path)
        body_limit = self._max_chars - len(header) - 1
        parts = self._split_with_overlap(body, body_limit)
        chunks: list[IndexChunk] = []
        for offset, (start, end) in enumerate(parts):
            blocks = self._blocks_in_range(spans, start, end)
            block_ids = tuple(block.block_id for block in blocks)
            methods = tuple(dict.fromkeys(block.extraction_method for block in blocks))
            page_start = min(block.page_start for block in blocks)
            page_end = max(block.page_end for block in blocks)
            metadata = {
                "page_start": page_start,
                "page_end": page_end,
                "content_type": content_type.value,
                "section_path": section_path,
                "extraction_methods": methods,
                "block_ids": block_ids,
                "warnings": document.warnings,
            }
            split_ordinal = first_split_ordinal + offset
            digest = hashlib.sha256(
                "\x1f".join((document.document_id, *block_ids, str(split_ordinal))).encode("utf-8")
            ).hexdigest()
            chunks.append(
                IndexChunk(
                    chunk_id=f"chunk-{digest}",
                    document_id=document.document_id,
                    filename=document.filename,
                    content_type=content_type,
                    text=f"{header}\n{body[start:end]}",
                    page_start=page_start,
                    page_end=page_end,
                    section_path=section_path,
                    metadata=metadata,
                    extraction_method=methods[0] if len(methods) == 1 else "mixed",
                )
            )
        return chunks

    def _body_from_blocks(
        self, blocks: Sequence[ContentBlock]
    ) -> tuple[str, tuple[tuple[int, int, ContentBlock], ...]]:
        parts: list[str] = []
        spans: list[tuple[int, int, ContentBlock]] = []
        cursor = 0
        for index, block in enumerate(blocks):
            start = cursor
            text = self._block_text(block)
            parts.append(text)
            cursor += len(text)
            if index < len(blocks) - 1:
                parts.append("\n\n")
                cursor += 2
            spans.append((start, cursor, block))
        return "".join(parts), tuple(spans)

    @staticmethod
    def _blocks_in_range(
        spans: Sequence[tuple[int, int, ContentBlock]], start: int, end: int
    ) -> tuple[ContentBlock, ...]:
        blocks: list[ContentBlock] = []
        seen_ids: set[str] = set()
        for span_start, span_end, block in spans:
            if start < span_end and end > span_start and block.block_id not in seen_ids:
                blocks.append(block)
                seen_ids.add(block.block_id)
        if not blocks:
            raise ValueError("chunk split did not include source content")
        return tuple(blocks)

    def _header(self, retrieval_label: str, section_path: tuple[str, ...]) -> str:
        budget = self._max_chars - 2
        section = " > ".join(section_path) or "(root)"
        full = f"[{retrieval_label}] Section: {section}"
        if len(full) <= budget:
            return full
        compact = f"[{retrieval_label}] Section: …"
        return compact if len(compact) <= budget else f"[{retrieval_label}]"

    def _split_with_overlap(self, text: str, limit: int) -> tuple[tuple[int, int], ...]:
        """Return bounded source ranges with effective overlap and guaranteed progress."""
        if len(text) <= limit:
            return ((0, len(text)),)
        if self._overlap_chars and limit < 2:
            raise ValueError("max_chars leaves no room for positive body overlap")

        overlap = min(self._overlap_chars, max(0, limit - 1))

        parts: list[tuple[int, int]] = []
        start = 0
        while start < len(text):
            end = min(start + limit, len(text))
            if end < len(text):
                boundary = max(
                    text.rfind("\n", start + 1, end),
                    text.rfind(" ", start + 1, end),
                )
                if boundary >= start + overlap:
                    end = boundary + 1
            if end - start <= overlap and end < len(text):
                end = min(start + limit, len(text))
            parts.append((start, end))
            if end == len(text):
                break
            start = end - overlap
        return tuple(parts)

    def _figure_body(
        self, figure: ContentBlock, captions: Sequence[ContentBlock]
    ) -> tuple[str, tuple[tuple[int, int, ContentBlock], ...]]:
        if not captions:
            return self._body_from_blocks((figure,))

        caption_text, caption_spans = self._body_from_blocks(captions)
        caption_prefix = "Caption: "
        description_prefix = "\n\nDescription: "
        figure_text = self._block_text(figure)
        body = f"{caption_prefix}{caption_text}{description_prefix}{figure_text}"
        spans = [
            (
                0 if index == 0 else start + len(caption_prefix),
                end + len(caption_prefix),
                block,
            )
            for index, (start, end, block) in enumerate(caption_spans)
        ]
        spans.append(
            (len(caption_prefix) + len(caption_text), len(body), figure)
        )
        return body, tuple(spans)
