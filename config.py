from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "db"

LLM_MODEL = "llama3:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"

COLLECTION_NAME = "sdk_docs"

CHUNK_SIZE = 750
CHUNK_OVERLAP = 120

TOP_K = 12

# If retrieved chunks are weaker than this, reject the question
RELEVANCE_THRESHOLD = 0.45