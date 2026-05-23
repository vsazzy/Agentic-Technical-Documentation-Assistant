from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "db"
LOGS_DIR = BASE_DIR / "logs"
EVAL_DIR = BASE_DIR / "eval"

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