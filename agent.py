import json
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import LLM_MODEL
from rag import format_context, generate_answer_from_context
from tools import (
    calculate_retrieval_metrics,
    citation_validator_tool,
    refusal_tool,
    retrieve_docs_tool,
    source_summary_tool,
)

PLANNER_PROMPT = """
You are a strict planner for a local SDK documentation assistant.

Your task is to classify the user's question.

Allowed intents:
- sdk_question: user asks about SDK docs, APIs, setup, hardware, embedded systems, errors, functions, configuration.
- source_lookup: user asks where something is found in the docs.
- summarization: user asks to summarize part of the SDK documentation.
- comparison: user asks to compare SDK APIs, modules, setup flows, or hardware features.
- out_of_scope: unrelated to the SDK documentation.

Important:
- CSS, HTML, general coding, career advice, life advice, and unrelated programming questions are out_of_scope.
- If unsure, choose sdk_question only if the question could reasonably belong to hardware SDK documentation.

Return ONLY valid JSON with this schema:
{{
  "intent": "sdk_question | source_lookup | summarization | comparison | out_of_scope",
  "needs_retrieval": true,
  "decision": "short decision",
  "reason": "short reason"
}}

User question:
{question}
"""

def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Safely parse model JSON output.
    If parsing fails, fall back to sdk_question so retrieval guardrails still protect the system.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "intent": "sdk_question",
            "needs_retrieval": True,
            "decision": "fallback_to_retrieval",
            "reason": "Planner output was not valid JSON, so the system fell back to retrieval.",
        }


def plan_query(question: str) -> Dict[str, Any]:
    """
    Agent planner.
    Uses local LLM to classify the query before deciding which tool to call.
    """
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_PROMPT),
            ("human", "{question}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"question": question})

    plan = safe_json_loads(response.content.strip())

    allowed_intents = {
        "sdk_question",
        "source_lookup",
        "summarization",
        "comparison",
        "out_of_scope",
    }

    if plan.get("intent") not in allowed_intents:
        plan["intent"] = "sdk_question"
        plan["decision"] = "fallback_to_retrieval"
        plan["reason"] = "Planner returned unknown intent."

    return plan


def build_structured_response(
    answer: str,
    intent: str,
    planner: Dict[str, Any],
    docs: List[Document],
    source_records: List[Dict[str, Any]],
    refused: bool,
    failure_reason: str | None,
) -> Dict[str, Any]:
    """
    Builds a structured JSON-like response for the UI and observability.
    """
    readable_sources = source_summary_tool(docs)
    retrieval_metrics = calculate_retrieval_metrics(source_records)

    return {
        "answer": answer,
        "intent": intent,
        "planner": planner,
        "refused": refused,
        "failure_reason": failure_reason,
        "sources": readable_sources,
        "source_records": source_records,
        "retrieval": retrieval_metrics,
        "model": LLM_MODEL,
    }


def run_agent(question: str) -> Dict[str, Any]:
    """
    Main agentic workflow.

    Workflow:
    1. Plan query intent.
    2. Refuse obvious out-of-scope questions.
    3. Retrieve SDK docs using retrieval tool.
    4. Validate source citations.
    5. Generate answer from context.
    6. Return structured response.
    """
    planner = plan_query(question)
    intent = planner.get("intent", "sdk_question")

    if intent == "out_of_scope":
        result = refusal_tool("Planner classified the query as out of scope.")
        result["planner"] = planner
        result["model"] = LLM_MODEL
        return result

    docs, source_records = retrieve_docs_tool(question)

    validation = citation_validator_tool(docs, source_records)

    if not validation["valid"]:
        result = refusal_tool(validation["reason"])
        result["planner"] = planner
        result["source_records"] = source_records
        result["retrieval"] = calculate_retrieval_metrics(source_records)
        result["model"] = LLM_MODEL
        return result

    context = format_context(docs)
    answer = generate_answer_from_context(question=question, context=context)

    # Final prompt-level safety check
    unsafe_phrases = [
        "based on general knowledge",
        "outside the documentation",
        "outside the provided documentation",
        "in general",
        "generally speaking",
        "from my knowledge",
    ]

    if any(phrase in answer.lower() for phrase in unsafe_phrases):
        result = refusal_tool("Answer appeared to use outside knowledge.")
        result["planner"] = planner
        result["source_records"] = source_records
        result["retrieval"] = calculate_retrieval_metrics(source_records)
        result["model"] = LLM_MODEL
        return result

    return build_structured_response(
        answer=answer,
        intent=intent,
        planner=planner,
        docs=docs,
        source_records=source_records,
        refused=False,
        failure_reason=None,
    )