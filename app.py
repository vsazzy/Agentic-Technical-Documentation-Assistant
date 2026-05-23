import streamlit as st

from agent import run_agent
from config import LLM_MODEL, EMBEDDING_MODEL, DOCS_DIR, DB_DIR
from observability import build_observability_event, elapsed_ms, log_event, start_timer


st.set_page_config(
    page_title="Local SDK Agent",
    page_icon="🧠",
    layout="wide",
)


def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Local SDK Agent")
        st.write("Private agentic RAG assistant for SDK documentation.")

        st.divider()

        st.subheader("Models")
        st.write(f"LLM: `{LLM_MODEL}`")
        st.write(f"Embeddings: `{EMBEDDING_MODEL}`")

        st.subheader("Folders")
        st.write(f"Docs: `{DOCS_DIR.name}/`")
        st.write(f"Vector DB: `{DB_DIR.name}/`")

        st.divider()

        st.subheader("Workflow")
        st.code(
            "uv run python ingest.py --reset\n"
            "uv run streamlit run app.py\n"
            "uv run streamlit run dashboard.py",
            language="bash",
        )

        st.divider()

        st.subheader("Agent Features")
        st.markdown(
            """
            - Planner
            - Retrieval tool
            - Citation validation
            - Refusal logic
            - Structured response
            - Observability logging
            """
        )

        st.divider()

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_agent_debug(result: dict) -> None:
    with st.expander("Agent trace"):
        planner = result.get("planner", {})

        st.subheader("Planner")
        st.json(planner)

        st.subheader("Retrieval")
        st.json(result.get("retrieval", {}))

        st.subheader("Source Records")
        st.json(result.get("source_records", []))

    sources = result.get("sources", [])

    if sources:
        with st.expander("Sources used"):
            for source in sources:
                st.write(f"- {source}")


def main() -> None:
    initialize_session_state()
    render_sidebar()

    st.title("Local SDK Agent")
    st.caption(
        "A privacy-preserving agentic RAG assistant for hardware SDK documentation. "
        "Runs locally using Ollama, ChromaDB, LangChain, and Streamlit."
    )

    render_chat_history()

    question = st.chat_input(
        "Ask about SDK setup, APIs, examples, errors, GPIO, UART, SPI, I2C..."
    )

    if not question:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        timer = start_timer()

        try:
            with st.spinner("Planning, retrieving, validating sources, and generating answer..."):
                result = run_agent(question)

            latency = elapsed_ms(timer)

            answer = result.get(
                "answer",
                "I could not find this in the provided SDK documentation.",
            )

            st.markdown(answer)

            render_agent_debug(result)

            event = build_observability_event(
                question=question,
                result=result,
                latency_ms=latency,
                tool_call_success=True,
            )

            log_event(event)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except Exception as exc:
            latency = elapsed_ms(timer)

            error_answer = (
                "Something went wrong while running the local SDK agent.\n\n"
                f"Error: `{exc}`"
            )

            st.error(error_answer)

            fallback_result = {
                "answer": error_answer,
                "intent": "error",
                "planner": {},
                "model": LLM_MODEL,
                "refused": True,
                "failure_reason": "runtime_error",
                "retrieval": {},
            }

            event = build_observability_event(
                question=question,
                result=fallback_result,
                latency_ms=latency,
                tool_call_success=False,
                error=str(exc),
            )

            log_event(event)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_answer,
                }
            )


if __name__ == "__main__":
    main()