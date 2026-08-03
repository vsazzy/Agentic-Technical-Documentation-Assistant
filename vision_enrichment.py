"""Selective, page-grounded PDF vision enrichment using a local Ollama model."""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from config import OLLAMA_BASE_URL, VISION_MIN_NATIVE_TEXT_CHARS, VISION_MODEL
from document_models import ContentBlock, ContentType, NormalizedDocument


JsonObject = Mapping[str, Any]
VisionTransport = Callable[[str, Mapping[str, Any], float], JsonObject]


@dataclass(frozen=True)
class VisionDescription:
    """Validated structured content returned by the local vision model."""

    summary: str
    visible_text: tuple[str, ...]
    relationships: tuple[str, ...]
    steps: tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("vision response summary must be non-empty")
        object.__setattr__(self, "summary", self.summary.strip())
        for field_name in ("visible_text", "relationships", "steps"):
            values = getattr(self, field_name)
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"vision response {field_name} must be a list of strings")
            object.__setattr__(
                self,
                field_name,
                tuple(value.strip() for value in values),
            )
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("vision response confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", float(self.confidence))

    @classmethod
    def from_mapping(cls, value: object) -> "VisionDescription":
        expected_keys = {
            "summary",
            "visible_text",
            "relationships",
            "steps",
            "confidence",
        }
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ValueError("vision response must match the required JSON schema")
        try:
            return cls(
                summary=value["summary"],
                visible_text=value["visible_text"],
                relationships=value["relationships"],
                steps=value["steps"],
                confidence=value["confidence"],
            )
        except (TypeError, ValueError) as error:
            if isinstance(error, ValueError) and "vision response" in str(error):
                raise
            raise ValueError("vision response must match the required JSON schema") from error


@dataclass(frozen=True)
class PageProfile:
    """The native extraction signals used by the enrichment policy."""

    page_number: int
    native_text_chars: int
    has_image: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.page_number, int)
            or isinstance(self.page_number, bool)
            or self.page_number < 1
        ):
            raise ValueError("page_number must be a one-based page number")
        if (
            not isinstance(self.native_text_chars, int)
            or isinstance(self.native_text_chars, bool)
            or self.native_text_chars < 0
        ):
            raise ValueError("native_text_chars must be non-negative")
        if not isinstance(self.has_image, bool):
            raise ValueError("has_image must be a boolean")


def should_enrich(
    page: PageProfile,
    *,
    min_native_text_chars: int = VISION_MIN_NATIVE_TEXT_CHARS,
) -> bool:
    """Select low-native-text or image-bearing pages for visual analysis."""
    if min_native_text_chars < 0:
        raise ValueError("min_native_text_chars must be non-negative")
    return page.has_image or page.native_text_chars < min_native_text_chars


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _post_json(url: str, payload: Mapping[str, Any], timeout: float) -> JsonObject:
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    # Ignore environment proxy settings so page content cannot leave loopback,
    # and reject redirects so a local endpoint cannot forward it elsewhere.
    with build_opener(ProxyHandler({}), _RejectRedirects()).open(
        request, timeout=timeout
    ) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, Mapping):
        raise ValueError("vision response envelope must be a JSON object")
    return result


class VisionClient:
    """A structured Ollama chat client restricted to loopback endpoints."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        transport: VisionTransport | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = _validate_local_base_url(base_url)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("vision model must be non-empty")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("vision timeout must be positive")
        self._model = model.strip()
        self._transport = transport or _post_json
        self._timeout = float(timeout)

    @classmethod
    def local(cls, *, timeout: float = 60.0) -> "VisionClient":
        """Build a client from the configured local Ollama endpoint and model."""
        return cls(OLLAMA_BASE_URL, VISION_MODEL, timeout=timeout)

    @staticmethod
    def response_schema() -> dict[str, Any]:
        string_array = {"type": "array", "items": {"type": "string"}}
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "visible_text": string_array,
                "relationships": string_array,
                "steps": string_array,
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "summary",
                "visible_text",
                "relationships",
                "steps",
                "confidence",
            ],
            "additionalProperties": False,
        }

    def describe(self, page_png: bytes, nearby_text: str) -> VisionDescription:
        """Describe one rendered page without sending data beyond loopback."""
        if not isinstance(page_png, bytes) or not page_png:
            raise ValueError("page_png must be non-empty bytes")
        if not isinstance(nearby_text, str):
            raise ValueError("nearby_text must be a string")
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Describe this PDF page for retrieval. Use the nearby native "
                        "text only as context; do not replace or transcribe it wholesale.\n\n"
                        f"Nearby native text:\n{nearby_text}"
                    ),
                    "images": [base64.b64encode(page_png).decode("ascii")],
                }
            ],
            "stream": False,
            "format": self.response_schema(),
        }
        response = self._transport(
            f"{self._base_url}/api/chat",
            payload,
            self._timeout,
        )
        try:
            message = response["message"]
            if not isinstance(message, Mapping):
                raise TypeError
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError
            parsed = json.loads(content)
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("vision response must contain structured JSON content") from error
        return VisionDescription.from_mapping(parsed)


def _validate_local_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Ollama base URL must be a local loopback URL")
    parsed = urlsplit(base_url.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Ollama base URL must be a local loopback URL")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("Ollama base URL must be a local loopback URL") from error

    hostname = parsed.hostname.lower()
    if hostname != "localhost":
        try:
            if not ipaddress.ip_address(hostname).is_loopback:
                raise ValueError
        except ValueError as error:
            raise ValueError("Ollama base URL must be a local loopback URL") from error
    return base_url.strip().rstrip("/")


class PageRenderer(Protocol):
    def page_count(self, pdf_path: Path) -> int:
        """Return the number of PDF pages."""

    def render_page(self, pdf_path: Path, page_number: int) -> bytes:
        """Render one one-based PDF page as PNG bytes."""


class PdfPageRenderer:
    """Lazy PyMuPDF page renderer used by the production enricher."""

    def __init__(self, *, dpi: int = 144) -> None:
        if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0:
            raise ValueError("dpi must be a positive integer")
        self._dpi = dpi

    def page_count(self, pdf_path: Path) -> int:
        import fitz

        with fitz.open(Path(pdf_path)) as pdf:
            return pdf.page_count

    def render_page(self, pdf_path: Path, page_number: int) -> bytes:
        import fitz

        with fitz.open(Path(pdf_path)) as pdf:
            if page_number < 1 or page_number > pdf.page_count:
                raise ValueError(f"PDF page {page_number} is out of range")
            pixmap = pdf.load_page(page_number - 1).get_pixmap(dpi=self._dpi, alpha=False)
            return pixmap.tobytes("png")


class VisionDescriber(Protocol):
    def describe(self, page_png: bytes, nearby_text: str) -> VisionDescription:
        """Describe a rendered page."""


class VisionEnricher:
    """Append local vision figures while preserving every native block."""

    def __init__(
        self,
        client: VisionDescriber,
        renderer: PageRenderer | None = None,
        *,
        min_native_text_chars: int = VISION_MIN_NATIVE_TEXT_CHARS,
    ) -> None:
        if min_native_text_chars < 0:
            raise ValueError("min_native_text_chars must be non-negative")
        self._client = client
        self._renderer = renderer or PdfPageRenderer()
        self._min_native_text_chars = min_native_text_chars

    def enrich(self, document: NormalizedDocument, pdf_path: Path) -> NormalizedDocument:
        """Append page-grounded figures and isolate failures to page warnings."""
        warnings = list(document.warnings)
        try:
            rendered_page_count = self._renderer.page_count(Path(pdf_path))
            if (
                not isinstance(rendered_page_count, int)
                or isinstance(rendered_page_count, bool)
                or rendered_page_count < 0
            ):
                raise ValueError("renderer returned an invalid page count")
        except Exception as error:
            warnings.append(
                "vision enrichment failed before page selection: "
                f"{_warning_detail(error)}"
            )
            return NormalizedDocument(
                document.document_id,
                document.filename,
                document.blocks,
                tuple(warnings),
            )

        total_pages = max(document.stats.pages, rendered_page_count)
        figures: list[ContentBlock] = []
        for page_number in range(1, total_pages + 1):
            page_blocks = tuple(
                block
                for block in document.blocks
                if block.page_start <= page_number <= block.page_end
            )
            if any(block.content_type is ContentType.FIGURE for block in page_blocks):
                continue
            profile = PageProfile(
                page_number=page_number,
                native_text_chars=sum(
                    len(block.text)
                    for block in page_blocks
                    if block.content_type in {ContentType.TEXT, ContentType.TABLE}
                ),
                has_image=any(
                    block.content_type is ContentType.IMAGE for block in page_blocks
                ),
            )
            if not should_enrich(
                profile,
                min_native_text_chars=self._min_native_text_chars,
            ):
                continue

            try:
                page_png = self._renderer.render_page(Path(pdf_path), page_number)
                nearby_text = "\n\n".join(
                    block.text
                    for block in page_blocks
                    if block.content_type is not ContentType.FIGURE
                )
                description = self._client.describe(page_png, nearby_text)
                figures.append(
                    _figure_block(
                        document=document,
                        page_number=page_number,
                        page_blocks=page_blocks,
                        description=description,
                    )
                )
            except Exception as error:
                warnings.append(
                    f"vision enrichment failed on page {page_number}: "
                    f"{_warning_detail(error)}"
                )

        return NormalizedDocument(
            document.document_id,
            document.filename,
            (*document.blocks, *figures),
            tuple(warnings),
        )


def _figure_block(
    *,
    document: NormalizedDocument,
    page_number: int,
    page_blocks: Sequence[ContentBlock],
    description: VisionDescription,
) -> ContentBlock:
    text_parts = [description.summary]
    for heading, values in (
        ("Visible text", description.visible_text),
        ("Relationships", description.relationships),
        ("Steps", description.steps),
    ):
        if values:
            text_parts.append(f"{heading}: " + "; ".join(values))

    section_path = next(
        (block.section_path for block in page_blocks if block.section_path),
        _preceding_section_path(document.blocks, page_number),
    )
    digest = hashlib.sha256(
        f"{document.document_id}|ollama_vision|{page_number}".encode("utf-8")
    ).hexdigest()
    return ContentBlock(
        block_id=f"vision-{digest}",
        content_type=ContentType.FIGURE,
        text="\n".join(text_parts),
        page_start=page_number,
        page_end=page_number,
        section_path=section_path,
        metadata={
            "source_page": page_number,
            "source_block_ids": tuple(block.block_id for block in page_blocks),
            "visible_text": description.visible_text,
            "relationships": description.relationships,
            "steps": description.steps,
            "confidence": description.confidence,
        },
        extraction_method="ollama_vision",
    )


def _preceding_section_path(
    blocks: Sequence[ContentBlock], page_number: int
) -> tuple[str, ...]:
    for block in reversed(blocks):
        if block.page_end < page_number and block.section_path:
            return block.section_path
    return ()


def _warning_detail(error: Exception) -> str:
    detail = " ".join(str(error).split())
    return detail or type(error).__name__
