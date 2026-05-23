from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    DB_DIR,
    COLLECTION_NAME,
    LLM_MODEL,
    EMBEDDING_MODEL,
    TOP_K,
    RELEVANCE_THRESHOLD
)


SYSTEM_PROMPT = """
You are a private local SDK documentation assistant.

You must answer ONLY from the provided documentation context.

Strict rules:
- Do NOT use general knowledge.
- Do NOT answer from memory.
- Do NOT provide examples unless they are supported by the context.
- Do NOT say "based on general knowledge".
- Do NOT continue with an outside answer after saying the docs do not contain it.
- If the answer is not clearly present in the context, respond exactly:
  "I could not find this in the provided SDK documentation."

Your domain:
- Hardware SDKs
- API references
- Embedded systems documentation
- Setup instructions
- SDK function usage
- Error codes
- Device-specific examples

Documentation context:
{context}
"""


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def get_vector_db() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def format_context(docs: List[Document]) -> str:
    formatted_chunks = []

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")

        if page is not None:
            source_label = f"{source}, page {page + 1}"
        else:
            source_label = source

        formatted_chunks.append(
            f"[Source {index}: {source_label}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def format_sources(docs: List[Document]) -> List[str]:
    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")

        if page is not None:
            sources.append(f"{source} — page {page + 1}")
        else:
            sources.append(source)

    return list(dict.fromkeys(sources))


def answer_question(question: str) -> Tuple[str, List[Document], List[str]]:
    """
    Retrieves relevant chunks and generates an answer.
    Includes a relevance guardrail to block out-of-document questions.
    """
    if not DB_DIR.exists():
        return (
            "Vector database not found. Please run `uv run python ingest.py --reset` first.",
            [],
            [],
        )

    vector_db = get_vector_db()

    docs_with_scores = vector_db.similarity_search_with_relevance_scores(
        query=question,
        k=TOP_K,
    )

    if not docs_with_scores:
        return (
            "I could not find this in the provided SDK documentation.",
            [],
            [],
        )

    filtered_docs = [
        doc for doc, score in docs_with_scores
        if score >= RELEVANCE_THRESHOLD
    ]

    if not filtered_docs:
        return (
            "I could not find this in the provided SDK documentation.",
            [],
            [],
        )

    context = format_context(filtered_docs)

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.0,
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    answer = response.content.strip()

    blocked_phrases = [
        "based on general knowledge",
        "however, based on",
        "outside the provided",
        "in general",
        "generally",
    ]

    if any(phrase in answer.lower() for phrase in blocked_phrases):
        answer = "I could not find this in the provided SDK documentation."

    sources = format_sources(filtered_docs)

    return answer, filtered_docs, sources