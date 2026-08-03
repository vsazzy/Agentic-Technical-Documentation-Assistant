import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "db"
LOGS_DIR = BASE_DIR / "logs"
EVAL_DIR = BASE_DIR / "eval"

MANAGED_DOCS_DIR = DOCS_DIR / "managed"
STAGING_DOCS_DIR = DOCS_DIR / "staging"
REGISTRY_FILE = DB_DIR / "document_registry.sqlite3"
ACTIVE_INDEX_FILE = DB_DIR / "active_index.json"

VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(200 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "2000"))
VISION_MIN_NATIVE_TEXT_CHARS = int(os.getenv("VISION_MIN_NATIVE_TEXT_CHARS", "120"))

LLM_MODEL = "llama3:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"

COLLECTION_NAME = "sdk_docs"

CHUNK_SIZE = 750
CHUNK_OVERLAP = 120

TOP_K = 15

# Guardrail threshold.
# If all retrieved chunks are below this score, refuse the answer.
RELEVANCE_THRESHOLD = 0.45

# Agent settings
ENABLE_AGENT_PLANNER = True

# Observability
LOG_FILE = LOGS_DIR / "rag_logs.jsonl"
