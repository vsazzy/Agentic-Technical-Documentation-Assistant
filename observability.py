import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

from config import LOG_FILE, LOGS_DIR


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def estimate_tokens(text: str) -> int:
    """
    Simple token estimate.
    Rough approximation: 1 token ≈ 4 characters.
    """
    if not text:
        return 0

    return max(1, len(text) // 4)


def start_timer() -> float:
    return time.perf_counter()


def elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def log_event(event: Dict[str, Any]) -> None:
    """
    Append one JSON event to logs/rag_logs.jsonl.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def build_observability_event(
    question: str,
    result: Dict[str, Any],
    latency_ms: int,
    tool_call_success: bool,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    answer = result.get("answer", "")

    retrieval = result.get("retrieval", {})
    planner = result.get("planner", {})

    return {
        "timestamp": now_iso(),
        "question": question,
        "answer_preview": answer[:300],
        "intent": result.get("intent"),
        "planner_decision": planner.get("decision"),
        "planner_reason": planner.get("reason"),
        "model": result.get("model"),
        "latency_ms": latency_ms,
        "refused": result.get("refused", False),
        "failure_reason": result.get("failure_reason"),
        "num_sources": retrieval.get("num_sources"),
        "retrieval_top_score": retrieval.get("top_score"),
        "retrieval_avg_score": retrieval.get("avg_score"),
        "tool_call_success": tool_call_success,
        "input_chars": len(question),
        "output_chars": len(answer),
        "approx_input_tokens": estimate_tokens(question),
        "approx_output_tokens": estimate_tokens(answer),
        "error": error,
    }