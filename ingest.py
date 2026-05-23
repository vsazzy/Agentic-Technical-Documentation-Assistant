import argparse
import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config import (
    DOCS_DIR,
    DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


SUPPORTED_TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".html",
    ".htm",
    ".csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest SDK documents into local Chroma DB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing vector database before ingesting.",
    )
    return parser.parse_args()


def load_documents() -> List[Document]:
    documents: List[Document] = []

    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Docs directory not found: {DOCS_DIR}")

    files = [path for path in DOCS_DIR.rglob("*") if path.is_file() and path.name != ".gitkeep"]

    if not files:
        raise ValueError(f"No files found in {DOCS_DIR}. Add SDK docs first.")

    for file_path in files:
        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                loaded_docs = loader.load()

            elif suffix in SUPPORTED_TEXT_EXTENSIONS:
                loader = TextLoader(str(file_path), encoding="utf-8")
                loaded_docs = loader.load()

            else:
                print(f"Skipping unsupported file: {file_path}")
                continue

            relative_path = file_path.relative_to(DOCS_DIR)

            for doc in loaded_docs:
                doc.metadata["source"] = str(relative_path)
                doc.metadata["file_type"] = suffix

            documents.extend(loaded_docs)
            print(f"Loaded: {relative_path}")

        except Exception as exc:
            print(f"Could not load {file_path}: {exc}")

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            "### ",
            "## ",
            "# ",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents(documents)


def create_vector_db(chunks: List[Document]) -> None:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION_NAME,
    )


def main() -> None:
    args = parse_args()

    if args.reset and DB_DIR.exists():
        print(f"Deleting existing vector database: {DB_DIR}")
        shutil.rmtree(DB_DIR)

    print("\nStarting local SDK document ingestion...\n")

    documents = load_documents()

    if not documents:
        raise ValueError("No supported documents were loaded.")

    print(f"\nLoaded document units: {len(documents)}")

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created from the documents.")

    print(f"Created chunks: {len(chunks)}")

    create_vector_db(chunks)

    print("\nIngestion complete.")
    print(f"Vector database saved to: {DB_DIR}")
    print("\nNow run:")
    print("uv run streamlit run app.py\n")


if __name__ == "__main__":
    main()