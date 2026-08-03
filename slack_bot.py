"""Slack Socket Mode interface for local RAG queries and PDF management."""

from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agent import run_agent
from document_store import DocumentStore
from observability import build_observability_event, elapsed_ms, log_event, start_timer
from slack_documents import (
    SlackDocumentError,
    SlackDocumentService,
    download_private_pdf,
    format_document_list,
    is_management_channel,
    parse_management_request,
    select_pdf_file,
)

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")
RAG_MANAGEMENT_CHANNEL_ID = os.getenv("RAG_MANAGEMENT_CHANNEL_ID", "")
ALLOWED_CHANNELS = os.getenv("ALLOWED_CHANNELS", "").strip()
DEFAULT_RESPONSE_TYPE = os.getenv("SLACK_RESPONSE_TYPE", "in_channel")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("slack-rag-bot")
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="slack-rag")


def get_allowed_channels() -> List[str]:
    return [channel.strip() for channel in ALLOWED_CHANNELS.split(",") if channel.strip()]


def is_channel_allowed(channel_id: str) -> bool:
    allowed = get_allowed_channels()
    return not allowed or channel_id in allowed


def normalize_sources(sources: Any) -> List[str]:
    if not sources:
        return []
    if not isinstance(sources, list):
        return [str(sources)]
    clean: list[str] = []
    for source in sources:
        if isinstance(source, str):
            clean.append(source)
        elif isinstance(source, dict):
            name = source.get("source") or source.get("file") or "unknown source"
            page = source.get("page")
            clean.append(f"{name} — page {page}" if page is not None else str(name))
        else:
            clean.append(str(source))
    return list(dict.fromkeys(clean))


def format_sources_for_slack(sources: List[str]) -> str:
    if not sources:
        return ""
    return "\n\n*Sources:*\n" + "\n".join(f"• `{source}`" for source in sources)


def format_agent_trace(result: Dict[str, Any]) -> str:
    lines = [
        f"*Intent:* `{result.get('intent', 'unknown')}`",
        f"*Refused:* `{result.get('refused', False)}`",
    ]
    if result.get("failure_reason"):
        lines.append(f"*Failure reason:* `{result['failure_reason']}`")
    retrieval = result.get("retrieval", {})
    if isinstance(retrieval, dict):
        for key in ("num_sources", "retrieved_docs", "top_score", "avg_score"):
            if key in retrieval:
                lines.append(f"*{key}:* `{retrieval[key]}`")
    return "\n".join(lines)


def format_response_for_slack(result: Dict[str, Any], show_trace: bool = False) -> str:
    answer = result.get("answer", "I could not find this in the provided documentation.")
    text = f"*Answer:*\n{answer}{format_sources_for_slack(normalize_sources(result.get('sources', [])))}"
    if show_trace:
        text += f"\n\n*Agent trace:*\n{format_agent_trace(result)}"
    return text


def run_rag_with_logging(question: str) -> Dict[str, Any]:
    timer = start_timer()
    try:
        result = run_agent(question)
        log_event(build_observability_event(question=question, result=result, latency_ms=elapsed_ms(timer), tool_call_success=True))
        return result
    except Exception as exc:
        result = {
            "answer": f"Something went wrong while running the local agent.\n\nError: `{exc}`",
            "intent": "error", "planner": {}, "refused": True,
            "failure_reason": "runtime_error", "retrieval": {}, "sources": [], "source_records": [],
        }
        log_event(build_observability_event(question=question, result=result, latency_ms=elapsed_ms(timer), tool_call_success=False, error=str(exc)))
        logger.error("RAG runtime error: %s\n%s", exc, traceback.format_exc())
        return result


def _require_management_channel(channel_id: str, respond) -> bool:
    if is_management_channel(channel_id, RAG_MANAGEMENT_CHANNEL_ID):
        return True
    respond(response_type="ephemeral", text="PDF management is only available in the configured management channel.")
    return False


def _recent_pdf(client, *, channel_id: str, user_id: str, requested_name: str = "") -> Mapping[str, Any]:
    history = client.conversations_history(channel=channel_id, limit=25)
    candidates: list[Mapping[str, Any]] = []
    for message in history.get("messages", []):
        if message.get("user") != user_id:
            continue
        for file_info in message.get("files", []):
            name = str(file_info.get("name", ""))
            if name.casefold().endswith(".pdf") and (not requested_name or name.casefold() == requested_name.casefold()):
                candidates.append(file_info)
    if not candidates:
        raise SlackDocumentError("Upload the PDF in this channel, then run `/rag-add [filename.pdf]`.")
    return candidates[0]


def _confirmation_blocks(filename: str, token: str) -> list[dict[str, Any]]:
    return [{
        "type": "actions",
        "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "Confirm deletion"}, "style": "danger", "action_id": "rag_remove_confirm", "value": token},
            {"type": "button", "text": {"type": "plain_text", "text": "Cancel"}, "action_id": "rag_remove_cancel", "value": token},
        ],
        "block_id": f"remove:{filename}",
    }]


def register_handlers(slack_app: App, service: SlackDocumentService) -> None:
    """Register all handlers; kept explicit so importing this module needs no tokens."""

    @slack_app.command("/ask-sdk")
    def handle_ask_rag(ack, respond, command):
        ack()
        channel_id, user_id = command.get("channel_id", ""), command.get("user_id", "")
        raw_text = command.get("text", "").strip()
        if not is_channel_allowed(channel_id):
            respond(response_type="ephemeral", text="This RAG bot is not enabled in this channel.")
            return
        if not raw_text:
            respond(response_type="ephemeral", text="Usage: `/ask-sdk your documentation question`")
            return
        show_trace = raw_text.startswith("--debug ")
        question = raw_text.removeprefix("--debug ").strip()
        respond(response_type=DEFAULT_RESPONSE_TYPE, text=f"*Question from <@{user_id}>:*\n> {question}\n\nRunning the local RAG agent...")
        executor.submit(lambda: respond(response_type=DEFAULT_RESPONSE_TYPE, text=format_response_for_slack(run_rag_with_logging(question), show_trace)))

    @slack_app.command("/rag-list")
    def handle_rag_list(ack, respond, command):
        ack()
        if not _require_management_channel(command.get("channel_id", ""), respond):
            return
        respond(response_type="ephemeral", text=format_document_list(service.list_documents()))

    @slack_app.command("/rag-add")
    def handle_rag_add(ack, respond, command, client):
        ack()
        channel_id, user_id = command.get("channel_id", ""), command.get("user_id", "")
        if not _require_management_channel(channel_id, respond):
            return
        respond(response_type="ephemeral", text="PDF accepted. Downloading and indexing it locally...")
        requested = command.get("text", "").strip()
        def work() -> None:
            try:
                info = _recent_pdf(client, channel_id=channel_id, user_id=user_id, requested_name=requested)
                path = download_private_pdf(info, bot_token=SLACK_BOT_TOKEN, store=service.store)
                receipt = service.add_staged(path, uploader_id=user_id)
                respond(response_type=DEFAULT_RESPONSE_TYPE, text=f"Added `{receipt.filename}`: {receipt.pages} pages, {receipt.tables} tables, {receipt.figures} figures, {receipt.chunks} chunks.")
            except Exception as error:
                logger.exception("Slack PDF add failed")
                respond(response_type="ephemeral", text=f"Could not add the PDF: {error}")
        executor.submit(work)

    @slack_app.command("/rag-remove")
    def handle_rag_remove(ack, respond, command):
        ack()
        channel_id, user_id = command.get("channel_id", ""), command.get("user_id", "")
        if not _require_management_channel(channel_id, respond):
            return
        filename = command.get("text", "").strip()
        if not filename:
            respond(response_type="ephemeral", text="Usage: `/rag-remove exact-filename.pdf`")
            return
        try:
            pending = service.request_removal(filename, requester_id=user_id, channel_id=channel_id)
            respond(response_type="ephemeral", text=f"Permanently delete `{pending.filename}` and rebuild the local index?", blocks=_confirmation_blocks(pending.filename, pending.token))
        except Exception as error:
            respond(response_type="ephemeral", text=str(error))

    @slack_app.action("rag_remove_confirm")
    def handle_remove_confirm(ack, respond, body):
        ack()
        channel_id = body.get("channel", {}).get("id", "")
        user_id = body.get("user", {}).get("id", "")
        token = body.get("actions", [{}])[0].get("value", "")
        respond(replace_original=True, text="Deletion confirmed. Rebuilding the local index...", blocks=[])
        def work() -> None:
            try:
                receipt = service.confirm_removal(token, requester_id=user_id, channel_id=channel_id)
                respond(replace_original=True, text=f"PDF removed. Rebuilt {len(receipt.documents)} remaining document(s).", blocks=[])
            except Exception as error:
                logger.exception("Slack PDF removal failed")
                respond(replace_original=True, text=f"Removal failed; the original PDF was restored: {error}", blocks=[])
        executor.submit(work)

    @slack_app.action("rag_remove_cancel")
    def handle_remove_cancel(ack, respond, body):
        ack()
        try:
            service.cancel_removal(body.get("actions", [{}])[0].get("value", ""), requester_id=body.get("user", {}).get("id", ""), channel_id=body.get("channel", {}).get("id", ""))
            respond(replace_original=True, text="PDF deletion cancelled.", blocks=[])
        except Exception as error:
            respond(replace_original=True, text=str(error), blocks=[])

    @slack_app.event("app_mention")
    def handle_app_mention(event, say):
        channel_id, user_id = event.get("channel", ""), event.get("user", "")
        text = event.get("text", "").strip()
        files = event.get("files", [])
        management = parse_management_request(text, has_pdf=bool(files))
        if management and management[0] == "add":
            if not is_management_channel(channel_id, RAG_MANAGEMENT_CHANNEL_ID):
                say("PDF management is only available in the configured management channel.")
                return
            say("PDF accepted. Downloading and indexing it locally...")
            def work() -> None:
                try:
                    info = select_pdf_file(files)
                    path = download_private_pdf(info, bot_token=SLACK_BOT_TOKEN, store=service.store)
                    receipt = service.add_staged(path, uploader_id=user_id)
                    say(f"Added `{receipt.filename}`: {receipt.pages} pages and {receipt.chunks} searchable chunks.")
                except Exception as error:
                    logger.exception("Slack natural-language PDF add failed")
                    say(f"Could not add the PDF: {error}")
            executor.submit(work)
            return
        if not is_channel_allowed(channel_id):
            say("This RAG bot is not enabled in this channel.")
            return
        parts = text.split(maxsplit=1)
        question = parts[1].strip() if len(parts) > 1 else ""
        if not question:
            say("Ask me a documentation question after mentioning me.")
            return
        say("Running the local RAG agent...")
        executor.submit(lambda: say(format_response_for_slack(run_rag_with_logging(question))))


def create_app() -> App | None:
    """Return a configured Bolt app, or ``None`` when credentials are absent."""
    if not SLACK_BOT_TOKEN or not SLACK_SIGNING_SECRET:
        return None
    slack_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
    register_handlers(slack_app, SlackDocumentService())
    return slack_app


app = create_app()


def main() -> None:
    missing = [name for name, value in (("SLACK_BOT_TOKEN", SLACK_BOT_TOKEN), ("SLACK_APP_TOKEN", SLACK_APP_TOKEN), ("SLACK_SIGNING_SECRET", SLACK_SIGNING_SECRET), ("RAG_MANAGEMENT_CHANNEL_ID", RAG_MANAGEMENT_CHANNEL_ID)) if not value]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    assert app is not None
    logger.info("Starting Slack RAG bot in Socket Mode...")
    SocketModeHandler(app, SLACK_APP_TOKEN).start()


if __name__ == "__main__":
    main()
