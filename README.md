# Local SDK RAG Assistant

A private local Retrieval-Augmented Generation assistant for hardware SDK documentation.

This project lets you chat with SDK manuals, API references, code examples, and troubleshooting guides using a local LLM.

## Demo Document

For the public demo, this project uses the official Raspberry Pi Pico C/C++ SDK documentation PDF.

Private SDK files should be placed in:

```text
docs/private/
```

Public demo docs can be placed in:

```text
docs/public/
```

## Tech Stack

- Python
- uv
- Streamlit
- LangChain
- Ollama
- ChromaDB

## Models

```text
LLM: llama3:latest
Embeddings: nomic-embed-text:latest
```

## Setup

### 1. Create environment

```bash
uv venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv sync
```

Or install manually:

```bash
uv add streamlit langchain langchain-community langchain-ollama langchain-chroma langchain-text-splitters chromadb pypdf
```

### 3. Pull Ollama models

```bash
ollama pull llama3:latest
ollama pull nomic-embed-text:latest
```

### 4. Download public demo SDK PDF

```bash
mkdir -p docs/public

curl -L "https://pip.raspberrypi.com/documents/RP-009085-KB-raspberry-pi-pico-c-sdk.pdf" \
  -o docs/public/raspberry-pi-pico-c-sdk.pdf
```

### 5. Build vector database

```bash
uv run python ingest.py --reset
```

### 6. Run Streamlit app

```bash
uv run streamlit run app.py
```

## Slack bot and PDF management

The bot answers questions and lets every member of one configured Slack channel
manage the local PDF corpus. After Slack downloads a private upload, extraction,
vision analysis, embeddings, storage, and retrieval stay on this machine.

### Slack app configuration

Enable **Socket Mode** and create an app-level token with `connections:write`.
Add these bot token scopes:

```text
app_mentions:read
channels:history
chat:write
commands
files:read
groups:history       # also required when the management channel is private
```

Subscribe to the `app_mention` bot event. Create these slash commands (the
request URL is not used in Socket Mode):

```text
/ask-sdk
/rag-add
/rag-list
/rag-remove
```

Invite the bot to the question channels and to the dedicated management
channel. Copy `.env.example` to `.env`, then set the three Slack credentials and
the management channel ID:

```bash
cp .env.example .env
uv run python slack_bot.py
```

### Slack usage

```text
/ask-sdk How do I follow the installation walkthrough?
/rag-list
/rag-add manual.pdf
/rag-remove manual.pdf
```

For `/rag-add`, first upload the PDF in the management channel, then run the
command. The filename is optional; when provided it selects that exact recent
upload. You can also upload one PDF while mentioning the bot and write
`please add this PDF`.

Removal always presents **Confirm deletion** and **Cancel** buttons. Only the
requesting member can confirm, and confirmations expire after ten minutes. A
confirmed removal deletes the local PDF and rebuilds the index from every
remaining PDF. Approved Slack uploads live in `docs/managed/`; incomplete
downloads live briefly in `docs/staging/` and are never indexed.

The Streamlit demo uses the same active multimodal index:

```bash
uv run streamlit run app.py
```

## Example Questions

```text
How do I initialize GPIO in the Pico SDK?
How do I configure UART?
What APIs are available for I2C?
How do I use PWM?
What is the setup flow for a Pico C SDK project?
Give me an example of blinking an LED.
What functions are used for SPI communication?
```

## Privacy

- Local LLM runs through Ollama.
- Embeddings are generated locally.
- Vector database is stored locally in `db/`.
- Private SDK files inside `docs/private/` are ignored by Git.
## Project Structure

```text
local-sdk-agent/
├── app.py                     # Main Streamlit chat UI
├── ingest.py                  # Document ingestion pipeline
├── rag.py                     # Core RAG logic: vector DB, context formatting, LLM answer generation
├── agent.py                   # Agentic workflow: planner, routing, validation, structured response
├── tools.py                   # Agent tools: retrieval, citation validation, refusal, metrics
├── observability.py           # Logging utilities for latency, retrieval scores, tokens, failures
├── dashboard.py               # Streamlit observability dashboard
├── config.py                  # Central configuration for models, paths, chunking, thresholds
├── pyproject.toml             # uv project dependencies
├── uv.lock                    # Locked dependency versions
├── README.md                  # Project documentation
├── .gitignore                 # Prevents private docs, vector DB, env files from being committed
├── .python-version            # Python version used by uv
│
├── docs/
│   ├── public/                # Public demo SDK documentation
│   │   └── raspberry-pi-pico-c-sdk.pdf
│   └── private/               # Private SDK docs, ignored by Git
│       └── .gitkeep
│
├── db/                        # Local Chroma vector database, ignored by Git
│
├── logs/
│   └── rag_logs.jsonl         # Observability logs, ignored or optionally committed for demo
│
├── eval/
├── screenshots/
└── assets/
```
## Architecture

```mermaid
flowchart TD
    A["Hardware SDK Docs"] --> B["Ingestion Pipeline"]
    B --> C["Chunking + Metadata"]
    C --> D["Local Embeddings"]
    D --> E["Chroma Vector DB"]

    F["User Query"] --> G["Planner Agent"]
    G --> H{"Intent"}

    H -->|"SDK Question"| I["Retrieval Tool"]
    H -->|"Out of Scope"| J["Refusal Tool"]
    H -->|"Source Lookup"| K["Source Inspector"]

    I --> L["Retrieved Chunks"]
    K --> L

    L --> M["Citation Validator"]
    M --> N{"Valid Sources?"}

    N -->|"Yes"| O["Local LLM Answer"]
    N -->|"No"| J

    O --> P["Structured JSON Response"]
    J --> P

    P --> Q["Streamlit UI"]
    P --> R["Observability Logger"]
    R --> S["Metrics Dashboard"]
```
