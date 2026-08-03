"""Deterministic, structure-aware chunks for normalized PDF documents."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

from document_models import ContentBlock, ContentType, IndexChunk, NormalizedDocument


class DocumentChunker:
    """Render semantic document units without crossing section or page boundaries."""

    _ATOMIC_TYPES = frozenset({ContentType.TABLE, ContentType.IMAGE, ContentType.FIGURE})

    def __init__(self, max_chars: int, overlap_chars: int) -> None:
        if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
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
        """Build ordered chunks while retaining normalized provenance in every chunk."""
        source_blocks = {block.block_id: block for block in document.blocks}
        chunks: list[IndexChunk] = []
        text_run: list[ContentBlock] = []
        split_ordinal = 0

        def flush_text_run() -> None:
            nonlocal split_ordinal
            if not text_run:
                return
            for group in self._fit_text_run(text_run):
                emitted = self._emit_group(group, group[0].content_type, None)
                chunks.extend(
                    self._to_index_chunks(document, emitted, split_ordinal)
                )
                split_ordinal += len(emitted)
            text_run.clear()

        for block in document.blocks:
            if block.content_type in self._ATOMIC_TYPES:
                flush_text_run()
                body = self._figure_body(block, source_blocks) if block.content_type is ContentType.FIGURE else block.text
                emitted = self._emit_group((block,), block.content_type, body)
                chunks.extend(self._to_index_chunks(document, emitted, split_ordinal))
                split_ordinal += len(emitted)
                continue

            if text_run and not self._compatible(text_run[-1], block):
                flush_text_run()
            text_run.append(block)

        flush_text_run()
        return chunks

    def _fit_text_run(self, blocks: Sequence[ContentBlock]) -> Iterable[tuple[ContentBlock, ...]]:
        """Keep adjacent text together whenever the rendered chunk has room."""
        group: list[ContentBlock] = []
        for block in blocks:
            candidate = tuple((*group, block))
            if group and len(self._render(candidate, candidate[0].content_type, None)) > self._max_chars:
                yield tuple(group)
                group = [block]
            else:
                group.append(block)
        if group:
            yield tuple(group)

    @staticmethod
    def _compatible(previous: ContentBlock, current: ContentBlock) -> bool:
        return (
            previous.content_type is current.content_type
            and previous.section_path == current.section_path
            and current.page_start <= previous.page_end + 1
        )

    def _emit_group(
        self,
        blocks: Sequence[ContentBlock],
        content_type: ContentType,
        body: str | None,
    ) -> tuple[tuple[Sequence[ContentBlock], ContentType, str], ...]:
        rendered = self._render(blocks, content_type, body)
        header, body_text = rendered.split("\n", 1)
        available = self._max_chars - len(header) - 1
        pieces = self._split_with_overlap(body_text, available)
        return tuple((blocks, content_type, f"{header}\n{piece}") for piece in pieces)

    @staticmethod
    def _render(
        blocks: Sequence[ContentBlock], content_type: ContentType, body: str | None
    ) -> str:
        section = " > ".join(blocks[0].section_path) or "(root)"
        rendered_body = body if body is not None else "\n\n".join(block.text for block in blocks)
        return f"[{content_type.value.upper()}] Section: {section}\n{rendered_body}"

    def _split_with_overlap(self, text: str, limit: int) -> tuple[str, ...]:
        """Split only an oversized semantic unit and retain an exact text overlap."""
        if len(text) <= limit or limit <= 0:
            return (text,)

        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + limit, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", start + 1, end + 1), text.rfind(" ", start + 1, end + 1))
                if boundary > start:
                    end = boundary + 1
            pieces.append(text[start:end])
            if end == len(text):
                break
            next_start = end - self._overlap_chars
            start = next_start if next_start > start else end
        return tuple(pieces)

    @staticmethod
    def _figure_body(block: ContentBlock, source_blocks: dict[str, ContentBlock]) -> str:
        source_ids = block.metadata.get("source_block_ids", ())
        captions = [
            source_blocks[block_id].text
            for block_id in source_ids
            if isinstance(block_id, str) and block_id in source_blocks
        ]
        if not captions:
            return block.text
        return f"Caption: {' '.join(captions)}\n\nDescription: {block.text}"

    def _to_index_chunks(
        self,
        document: NormalizedDocument,
        emitted: Sequence[tuple[Sequence[ContentBlock], ContentType, str]],
        first_split_ordinal: int,
    ) -> list[IndexChunk]:
        chunks: list[IndexChunk] = []
        for offset, (blocks, content_type, text) in enumerate(emitted):
            block_ids = tuple(block.block_id for block in blocks)
            methods = tuple(dict.fromkeys(block.extraction_method for block in blocks))
            page_start = min(block.page_start for block in blocks)
            page_end = max(block.page_end for block in blocks)
            metadata = {
                "page_start": page_start,
                "page_end": page_end,
                "content_type": content_type.value,
                "section_path": blocks[0].section_path,
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
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    section_path=blocks[0].section_path,
                    metadata=metadata,
                    extraction_method=methods[0] if len(methods) == 1 else "mixed",
                )
            )
        return chunks
