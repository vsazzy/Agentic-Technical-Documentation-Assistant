import streamlit as st

from rag import answer_question
from config import LLM_MODEL, EMBEDDING_MODEL, DOCS_DIR, DB_DIR


st.set_page_config(
    page_title="Local SDK RAG Assistant",
    page_icon="🧠",
    layout="wide",
)


def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar() -> None:
    with st.sidebar:
        st.title("Local SDK RAG")
        st.write("Private documentation assistant running locally.")

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
            "uv run streamlit run app.py",
            language="bash",
        )

        st.divider()

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    initialize_session_state()
    render_sidebar()

    st.title("Local SDK Documentation Assistant")
    st.caption(
        "Ask questions about hardware SDK docs. "
        "Runs locally using Ollama, Chroma, LangChain, and Streamlit."
    )

    render_chat_history()

    question = st.chat_input("Ask about SDK setup, APIs, examples, errors, GPIO, UART, SPI, I2C...")

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
        with st.spinner("Searching local SDK docs..."):
            answer, retrieved_docs, sources = answer_question(question)

        st.markdown(answer)

        if sources:
            with st.expander("Sources used"):
                for source in sources:
                    st.write(f"- {source}")

        if retrieved_docs:
            with st.expander("Retrieved context preview"):
                for index, doc in enumerate(retrieved_docs, start=1):
                    source = doc.metadata.get("source", "unknown source")
                    page = doc.metadata.get("page")

                    if page is not None:
                        label = f"{source}, page {page + 1}"
                    else:
                        label = source

                    st.markdown(f"**Chunk {index}: {label}**")
                    st.code(doc.page_content[:1500])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


if __name__ == "__main__":
    main()