"""Slack-facing document lifecycle helpers.

The Slack transport lives in :mod:`slack_bot`; this module keeps document
management independently testable and contains no Slack SDK dependency.
"""

from __future__ import annotations

import os
import re
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from document_store import DocumentRecord, DocumentStore
from ingest import IngestionPipeline, IngestionReceipt, RebuildReceipt, build_ingestion_pipeline


ADD_RE = re.compile(r"\b(?:please\s+)?(?:add|ingest|index)\b.*\b(?:pdf|document|file)\b", re.I)
REMOVE_RE = re.compile(r"\b(?:remove|delete)\s+([^\n]+?\.pdf)\b", re.I)


class SlackDocumentError(RuntimeError):
    """A concise error safe to post back to Slack."""


@dataclass(frozen=True)
class PendingRemoval:
    token: str
    filename: str
    document_id: str
    requester_id: str
    channel_id: str
    expires_at: float


def is_management_channel(channel_id: str, management_channel_id: str) -> bool:
    return bool(management_channel_id and channel_id == management_channel_id)


def parse_management_request(text: str, *, has_pdf: bool = False) -> tuple[str, str | None] | None:
    """Recognize the deliberately small natural-language management surface."""
    normalized = " ".join((text or "").split())
    removal = REMOVE_RE.search(normalized)
    if removal:
        return "remove", Path(removal.group(1).strip(" `\"'")).name
    if has_pdf and ADD_RE.search(normalized):
        return "add", None
    return None


def select_pdf_file(files: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    candidates = [
        item
        for item in files
        if str(item.get("mimetype", "")).casefold() == "application/pdf"
        or str(item.get("name", "")).casefold().endswith(".pdf")
    ]
    if len(candidates) != 1:
        raise SlackDocumentError("Attach exactly one PDF to this request.")
    return candidates[0]


def download_private_pdf(
    file_info: Mapping[str, Any],
    *,
    bot_token: str,
    store: DocumentStore,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> Path:
    """Download one Slack private file with a bearer token and stage it safely."""
    url = str(file_info.get("url_private_download") or file_info.get("url_private") or "")
    filename = str(file_info.get("name") or "upload.pdf")
    if not url.startswith("https://"):
        raise SlackDocumentError("Slack did not provide a secure private download URL.")
    declared_size = int(file_info.get("size") or 0)
    if declared_size > store.max_pdf_bytes:
        raise SlackDocumentError("PDF exceeds the configured upload-size limit.")
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {bot_token}"})
    chunks: list[bytes] = []
    total = 0
    try:
        with opener(request, timeout=60) as response:
            while data := response.read(1024 * 1024):
                total += len(data)
                if total > store.max_pdf_bytes:
                    raise SlackDocumentError("PDF exceeds the configured upload-size limit.")
                chunks.append(data)
    except SlackDocumentError:
        raise
    except Exception as error:
        raise SlackDocumentError(f"Could not download the Slack PDF: {error}") from error
    return store.stage_bytes(filename, b"".join(chunks))


class SlackDocumentService:
    """Add, list, and safely remove PDFs using the production ingestion pipeline."""

    def __init__(
        self,
        *,
        store: DocumentStore | None = None,
        pipeline: IngestionPipeline | None = None,
        confirmation_ttl_seconds: int = 600,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store or DocumentStore()
        self.pipeline = pipeline or build_ingestion_pipeline(store=self.store)
        self.confirmation_ttl_seconds = confirmation_ttl_seconds
        self._clock = clock
        self._pending: dict[str, PendingRemoval] = {}
        self._lock = threading.RLock()

    def add_staged(self, staged_path: Path, *, uploader_id: str) -> IngestionReceipt:
        """Promote a validated staged PDF and index only that PDF."""
        with self._lock:
            previous = self.store.get_by_filename(Path(staged_path).name)
            managed_path = self.store.promote(staged_path)
            try:
                return self.pipeline.ingest_pdf(managed_path, uploader=uploader_id)
            except Exception:
                # register()/ingest_pdf() owns replacement rollback; for a brand-new
                # failed upload, remove an unregistered promoted file.
                if previous is None:
                    managed_path.unlink(missing_ok=True)
                raise

    def list_documents(self) -> list[DocumentRecord]:
        return self.store.list_active()

    def request_removal(
        self, filename: str, *, requester_id: str, channel_id: str
    ) -> PendingRemoval:
        record = self.store.get_by_filename(filename)
        if record is None or record.status != "active":
            raise SlackDocumentError(f"No active PDF named `{filename}` was found.")
        token = uuid.uuid4().hex
        pending = PendingRemoval(
            token=token,
            filename=record.filename,
            document_id=record.document_id,
            requester_id=requester_id,
            channel_id=channel_id,
            expires_at=self._clock() + self.confirmation_ttl_seconds,
        )
        self._pending[token] = pending
        return pending

    def cancel_removal(self, token: str, *, requester_id: str, channel_id: str) -> None:
        self._claim_pending(token, requester_id=requester_id, channel_id=channel_id)

    def confirm_removal(
        self, token: str, *, requester_id: str, channel_id: str
    ) -> RebuildReceipt:
        pending = self._claim_pending(
            token, requester_id=requester_id, channel_id=channel_id
        )
        with self._lock:
            record = self.store.get_by_filename(pending.filename)
            if record is None or record.document_id != pending.document_id:
                raise SlackDocumentError("The PDF changed since deletion was requested.")
            if len(self.store.discover_corpus()) <= 1:
                raise SlackDocumentError("Cannot remove the final PDF from the corpus.")
            original = Path(record.path)
            backup = self.store.staging_dir / f"delete-{token}-{record.filename}"
            os.replace(original, backup)
            try:
                receipt = self.pipeline.rebuild_corpus()
                self.store.mark_deleted(record.document_id)
            except Exception:
                os.replace(backup, original)
                raise
            backup.unlink(missing_ok=True)
            return receipt

    def _claim_pending(
        self, token: str, *, requester_id: str, channel_id: str
    ) -> PendingRemoval:
        pending = self._pending.get(token)
        if pending is None:
            raise SlackDocumentError("This deletion confirmation is no longer valid.")
        if pending.requester_id != requester_id or pending.channel_id != channel_id:
            raise SlackDocumentError("Only the original requester can confirm this deletion.")
        if pending.expires_at < self._clock():
            self._pending.pop(token, None)
            raise SlackDocumentError("This deletion confirmation has expired.")
        self._pending.pop(token, None)
        return pending


def format_document_list(records: list[DocumentRecord]) -> str:
    if not records:
        return "No PDFs are currently indexed."
    lines = [f"• `{record.filename}` — {record.page_count} pages" for record in records]
    return "*Indexed PDFs:*\n" + "\n".join(lines)
