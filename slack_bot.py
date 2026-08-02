# slack_bot.py

import os
import logging
import traceback
from typing import Any, Dict, List

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agent import run_agent
from observability import (
    build_observability_event,
    elapsed_ms,
    log_event,
    start_timer,
)


load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

ALLOWED_CHANNELS = os.getenv("ALLOWED_CHANNELS", "").strip()
DEFAULT_RESPONSE_TYPE = os.getenv("SLACK_RESPONSE_TYPE", "in_channel")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("slack-rag-bot")


missing = []

if not SLACK_BOT_TOKEN:
    missing.append("SLACK_BOT_TOKEN")

if not SLACK_APP_TOKEN:
    missing.append("SLACK_APP_TOKEN")

if not SLACK_SIGNING_SECRET:
    missing.append("SLACK_SIGNING_SECRET")

if missing:
    raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")


app = App(
    token=SLACK_BOT_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET,
)


def get_allowed_channels() -> List[str]:
    if not ALLOWED_CHANNELS:
        return []

    return [
        channel.strip()
        for channel in ALLOWED_CHANNELS.split(",")
        if channel.strip()
    ]


def is_channel_allowed(channel_id: str) -> bool:
    allowed_channels = get_allowed_channels()

    if not allowed_channels:
        return True

    return channel_id in allowed_channels


def normalize_sources(sources: Any) -> List[str]:
    if not sources:
        return []

    if isinstance(sources, list):
        clean_sources = []

        for source in sources:
            if isinstance(source, str):
                clean_sources.append(source)
            elif isinstance(source, dict):
                source_name = source.get("source") or source.get("file") or "unknown source"
                page = source.get("page")

                if page is not None:
                    clean_sources.append(f"{source_name} — page {page}")
                else:
                    clean_sources.append(str(source_name))
            else:
                clean_sources.append(str(source))

        return list(dict.fromkeys(clean_sources))

    return [str(sources)]


def format_sources_for_slack(sources: List[str]) -> str:
    if not sources:
        return ""

    source_lines = "\n".join([f"• `{source}`" for source in sources])
    return f"\n\n*Sources:*\n{source_lines}"


def format_agent_trace(result: Dict[str, Any]) -> str:
    intent = result.get("intent", "unknown")
    refused = result.get("refused", False)
    failure_reason = result.get("failure_reason")
    retrieval = result.get("retrieval", {})

    trace_lines = [
        f"*Intent:* `{intent}`",
        f"*Refused:* `{refused}`",
    ]

    if failure_reason:
        trace_lines.append(f"*Failure reason:* `{failure_reason}`")

    if isinstance(retrieval, dict) and retrieval:
        for key in ["retrieved_docs", "avg_score", "min_score", "max_score"]:
            if key in retrieval:
                trace_lines.append(f"*{key}:* `{retrieval[key]}`")

    return "\n".join(trace_lines)


def format_response_for_slack(result: Dict[str, Any], show_trace: bool = False) -> str:
    answer = result.get(
        "answer",
        "I could not find this in the provided SDK documentation.",
    )

    sources = normalize_sources(result.get("sources", []))

    text = f"*Answer:*\n{answer}"
    text += format_sources_for_slack(sources)

    if show_trace:
        text += f"\n\n*Agent trace:*\n{format_agent_trace(result)}"

    return text


def run_rag_with_logging(question: str) -> Dict[str, Any]:
    timer = start_timer()

    try:
        result = run_agent(question)
        latency = elapsed_ms(timer)

        event = build_observability_event(
            question=question,
            result=result,
            latency_ms=latency,
            tool_call_success=True,
        )
        log_event(event)

        return result

    except Exception as exc:
        latency = elapsed_ms(timer)

        fallback_result = {
            "answer": (
                "Something went wrong while running the local SDK agent.\n\n"
                f"Error: `{exc}`"
            ),
            "intent": "error",
            "planner": {},
            "refused": True,
            "failure_reason": "runtime_error",
            "retrieval": {},
            "sources": [],
            "source_records": [],
        }

        event = build_observability_event(
            question=question,
            result=fallback_result,
            latency_ms=latency,
            tool_call_success=False,
            error=str(exc),
        )
        log_event(event)

        logger.error("RAG runtime error: %s", str(exc))
        logger.error(traceback.format_exc())

        return fallback_result


@app.command("/ask-sdk")
def handle_ask_rag(ack, respond, command):
    ack()

    channel_id = command.get("channel_id", "")
    user_id = command.get("user_id", "")
    raw_text = command.get("text", "").strip()

    if not is_channel_allowed(channel_id):
        respond(
            response_type="ephemeral",
            text="This RAG bot is not enabled in this channel.",
        )
        return

    if not raw_text:
        respond(
            response_type="ephemeral",
            text=(
                "Ask a question like:\n"
                "`/ask-sdk how do I initialize GPIO in the Pico SDK?`"
            ),
        )
        return

    show_trace = False
    question = raw_text

    if raw_text.startswith("--debug "):
        show_trace = True
        question = raw_text.replace("--debug ", "", 1).strip()

    logger.info(
        "Slack query from user=%s channel=%s question=%s",
        user_id,
        channel_id,
        question,
    )

    respond(
        response_type=DEFAULT_RESPONSE_TYPE,
        text=(
        f"*Question from <@{user_id}>:*\n"
        f"> {question}\n\n"
        "Running the local RAG agent now..."
    ),
    )

    result = run_rag_with_logging(question)
    final_text = format_response_for_slack(result, show_trace=show_trace)

    respond(
        response_type=DEFAULT_RESPONSE_TYPE,
        text=final_text,
    )


@app.event("app_mention")
def handle_app_mention(event, say):
    channel_id = event.get("channel", "")
    user_id = event.get("user", "")
    text = event.get("text", "").strip()

    if not is_channel_allowed(channel_id):
        say("This RAG bot is not enabled in this channel.")
        return

    parts = text.split(maxsplit=1)
    question = parts[1].strip() if len(parts) > 1 else ""

    if not question:
        say("Ask me a documentation question after mentioning me.")
        return

    logger.info(
        "Slack mention from user=%s channel=%s question=%s",
        user_id,
        channel_id,
        question,
    )

    result = run_rag_with_logging(question)
    final_text = format_response_for_slack(result, show_trace=False)

    say(final_text)


if __name__ == "__main__":
    logger.info("Starting Slack RAG bot in Socket Mode...")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()