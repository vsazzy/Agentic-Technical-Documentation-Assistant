from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import RELEVANCE_THRESHOLD, TOP_K
from rag import get_vector_db, format_sources


def retrieve_docs_tool(question: str) -> Tuple[List[Document], List[Dict[str, Any]]]:
    """
    Tool: Retrieve relevant SDK documentation chunks from ChromaDB.

    Returns:
        filtered_docs: documents above relevance threshold
        source_records: structured source metadata with scores
    """
    vector_db: Chroma = get_vector_db()

    expanded_question = expand_sdk_query(question)

    docs_with_scores = vector_db.similarity_search_with_relevance_scores(
        query=expanded_question,
        k=TOP_K,
    )

    filtered_docs: List[Document] = []
    source_records: List[Dict[str, Any]] = []

    for doc, score in docs_with_scores:
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")

        record = {
            "source": source,
            "page": page + 1 if page is not None else None,
            "score": round(float(score), 4),
            "passed_threshold": bool(score >= RELEVANCE_THRESHOLD),
        }

        source_records.append(record)

        if score >= RELEVANCE_THRESHOLD:
            filtered_docs.append(doc)

    return filtered_docs, source_records


def citation_validator_tool(
    docs: List[Document],
    source_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Tool: Validate whether retrieved sources are strong enough to answer.

    A valid answer requires:
    - at least one retrieved document
    - at least one source above the relevance threshold
    """
    valid_sources = [
        source for source in source_records
        if source.get("passed_threshold") is True
    ]

    is_valid = len(docs) > 0 and len(valid_sources) > 0

    if not is_valid:
        return {
            "valid": False,
            "reason": "No retrieved source passed the relevance threshold.",
            "valid_sources": [],
        }

    return {
        "valid": True,
        "reason": "At least one retrieved source passed the relevance threshold.",
        "valid_sources": valid_sources,
    }


def refusal_tool(reason: str) -> Dict[str, Any]:
    """
    Tool: Return a consistent refusal response.
    """
    return {
        "answer": "I could not find this in the provided SDK documentation.",
        "intent": "out_of_scope",
        "refused": True,
        "failure_reason": reason,
        "sources": [],
        "retrieval": {
            "num_sources": 0,
            "top_score": None,
            "avg_score": None,
        },
    }


def source_summary_tool(docs: List[Document]) -> List[str]:
    """
    Tool: Return readable source labels.
    """
    return format_sources(docs)


def calculate_retrieval_metrics(source_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tool: Calculate retrieval score statistics.
    """
    if not source_records:
        return {
            "num_sources": 0,
            "top_score": None,
            "avg_score": None,
        }

    scores = [source["score"] for source in source_records]

    return {
        "num_sources": len(source_records),
        "top_score": max(scores),
        "avg_score": round(sum(scores) / len(scores), 4),
    }


def expand_sdk_query(question: str) -> str:
    """
    Expands common SDK questions with likely API/reference terms.
    This improves retrieval for hardware SDK documentation.
    """
    q = question.lower()

    expansions = []

    if "uart" in q:
        expansions.extend([
            "UART uart_init uart_set_baudrate uart_set_format uart_set_hw_flow gpio_set_function UART_TX UART_RX baud rate serial communication",
        ])

    if "gpio" in q:
        expansions.extend([
            "GPIO gpio_init gpio_set_dir gpio_put gpio_get gpio_set_function input output pin",
        ])

    if "i2c" in q:
        expansions.extend([
            "I2C i2c_init i2c_write_blocking i2c_read_blocking SDA SCL baudrate",
        ])

    if "spi" in q:
        expansions.extend([
            "SPI spi_init spi_write_blocking spi_read_blocking MOSI MISO SCK CS",
        ])

    if "pwm" in q:
        expansions.extend([
            "PWM pwm_config pwm_init pwm_set_gpio_level pwm_set_wrap pwm_set_clkdiv",
        ])

    if "pio" in q:
        expansions.extend([
            "PIO pio_sm_init pio_add_program pio_sm_set_enabled state machine instruction set JMP MOV SET IRQ",
        ])

    if not expansions:
        return question

    return question + "\n\nRelevant SDK/API terms:\n" + "\n".join(expansions)