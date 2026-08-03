# Local Multimodal PDF Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully local, structure-aware PDF ingestion pipeline that indexes text, tables, OCR, figures, screenshots, and walkthrough descriptions with page provenance in ChromaDB.

**Architecture:** Docling converts PDFs into typed content blocks, a local Ollama vision adapter selectively enriches visual pages, and a structure-aware chunker produces stable chunks. A registry/document store makes source PDFs authoritative while a versioned index manager supports incremental upsert and recoverable rebuilds.

**Tech Stack:** Python 3.11, pytest, Docling, Pydantic/dataclasses, Ollama HTTP API, LangChain documents, Ollama embeddings, ChromaDB, SQLite.

## Global Constraints

- OCR, visual analysis, embeddings, retrieval, and answer generation run locally.
- Existing PDFs at `docs/*.pdf` remain supported; managed PDFs live at `docs/managed/`; staging is excluded from corpus scans.
- Every indexed block retains stable document ID, filename, one-based page provenance, section path, content type, and extraction method.
- Adds are incremental and idempotent; removals rebuild a versioned index and switch only after verification.
- Failed work must not leave active partial files, registry records, or chunks.
- No document content, page image, or extracted text may be sent to a cloud service.

---

## File Structure

- `config.py`: paths, limits, local model names, registry and active-index configuration.
- `document_models.py`: immutable normalized blocks, documents, chunks, and extraction statistics.
- `document_store.py`: safe paths, hashing, validation, staging/promotion, registry transactions, corpus discovery.
- `pdf_extractor.py`: Docling adapter that creates normalized text/table/OCR blocks.
- `vision_enrichment.py`: local Ollama vision client and visual-page enrichment policy.
- `document_chunker.py`: semantic chunk construction and stable chunk IDs.
- `index_manager.py`: Chroma incremental upsert, versioned rebuild, verification, and active pointer.
- `ingest.py`: CLI composition root for rebuilding the complete corpus.
- `rag.py`, `tools.py`: active-index retrieval and enriched context/source formatting.
- `tests/fixtures/`: deterministic miniature PDFs and extractor/vision payloads.

### Task 1: Configuration and normalized document contracts

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.py`
- Create: `document_models.py`
- Create: `tests/test_document_models.py`

**Interfaces:**
- Produces: `ContentType`, `ContentBlock`, `NormalizedDocument`, `IndexChunk`, `ExtractionStats`; configuration paths and limits used by every later task.

- [ ] **Step 1: Add dependencies and write failing model tests**

Add `docling`, `pymupdf`, and `pytest` to `pyproject.toml`. Create tests asserting one-based pages, valid content types, deterministic serialization, and rejection of empty document IDs/content.

```python
def test_content_block_requires_one_based_page():
    with pytest.raises(ValueError, match="page_start"):
        ContentBlock(block_id="b1", content_type=ContentType.TEXT,
                     text="hello", page_start=0, page_end=1)

def test_normalized_document_counts_content_types():
    doc = NormalizedDocument(document_id="sha256:abc", filename="guide.pdf", blocks=[
        ContentBlock("b1", ContentType.TEXT, "intro", 1, 1),
        ContentBlock("b2", ContentType.TABLE, "|A|B|", 2, 2),
    ])
    assert doc.stats.tables == 1
    assert doc.stats.pages == 2
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `.venv/bin/pytest tests/test_document_models.py -v`

Expected: collection fails because `document_models` does not exist.

- [ ] **Step 3: Implement the typed contracts and configuration**

Use string enums and frozen dataclasses. `ContentBlock` fields are `block_id`, `content_type`, `text`, `page_start`, `page_end`, `section_path=()`, `metadata={}`, and `extraction_method="docling"`. `IndexChunk` adds `chunk_id`, `document_id`, and `filename`. Validate non-empty IDs/text and `1 <= page_start <= page_end` in `__post_init__`.

Add exact configuration names:

```python
MANAGED_DOCS_DIR = DOCS_DIR / "managed"
STAGING_DOCS_DIR = DOCS_DIR / "staging"
REGISTRY_FILE = DB_DIR / "document_registry.sqlite3"
ACTIVE_INDEX_FILE = DB_DIR / "active_index.json"
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(200 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "2000"))
VISION_MIN_NATIVE_TEXT_CHARS = int(os.getenv("VISION_MIN_NATIVE_TEXT_CHARS", "120"))
```

- [ ] **Step 4: Sync and run tests**

Run: `uv sync`

Run: `.venv/bin/pytest tests/test_document_models.py -v`

Expected: all model tests pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock config.py document_models.py tests/test_document_models.py
git commit -m "feat: add multimodal document contracts"
```

### Task 2: Safe document store and SQLite registry

**Files:**
- Create: `document_store.py`
- Create: `tests/test_document_store.py`

**Interfaces:**
- Consumes: configuration paths from Task 1.
- Produces: `DocumentRecord`, `DocumentStore.validate_pdf(path)`, `stage_bytes(filename, data)`, `promote(staged_path)`, `restore_backup(path)`, `discover_corpus()`, `register(record)`, `get_by_filename(name)`, `list_active()`, `mark_failed(document_id, error)`, and `mark_deleted(document_id)`.

- [ ] **Step 1: Write failing safety and registry tests**

Cover filename normalization, traversal rejection, `%PDF-` signature validation, size/page limits, SHA-256 duplicate detection, root/managed discovery excluding staging, promotion, and SQLite uniqueness.

```python
def test_stage_rejects_path_traversal(store):
    with pytest.raises(DocumentValidationError):
        store.stage_bytes("../../secret.pdf", b"%PDF-1.7\n")

def test_discover_corpus_excludes_staging(store, pdf_bytes):
    (store.docs_dir / "root.pdf").write_bytes(pdf_bytes)
    (store.managed_dir / "managed.pdf").write_bytes(pdf_bytes)
    (store.staging_dir / "partial.pdf").write_bytes(pdf_bytes)
    assert [p.name for p in store.discover_corpus()] == ["root.pdf", "managed.pdf"]
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `.venv/bin/pytest tests/test_document_store.py -v`

Expected: import fails because `document_store` does not exist.

- [ ] **Step 3: Implement validation, storage, and registry transactions**

Use `Path.resolve()` plus `is_relative_to()` for containment, `hashlib.sha256` for identity, PyMuPDF for page-count validation, `os.replace` for promotion, and SQLite uniqueness constraints on `document_id`, `sha256`, and active normalized filename. Create directories lazily. Never accept symlinks as upload targets.

Registry schema includes `documents`, `ingestion_jobs`, and `index_versions`. Use parameterized SQL and a connection context manager that commits on success and rolls back on failure.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_document_store.py -v`

Expected: all store and registry tests pass.

- [ ] **Step 5: Commit**

```bash
git add document_store.py tests/test_document_store.py
git commit -m "feat: add safe PDF store and registry"
```

### Task 3: Docling structured extraction adapter

**Files:**
- Create: `pdf_extractor.py`
- Create: `tests/test_pdf_extractor.py`
- Create: `tests/fixtures/docling_document.json`

**Interfaces:**
- Consumes: `ContentBlock`, `NormalizedDocument`, `DocumentStore.validate_pdf`.
- Produces: `PdfExtractor.extract(path: Path, document_id: str) -> NormalizedDocument` and injectable `DoclingBackend.convert(path)`.

- [ ] **Step 1: Write failing adapter tests with a fake backend**

The fixture must include a heading, paragraph, two-page table, list, OCR block, and figure caption. Assert reading order, section propagation, Markdown table preservation, one-based pages, and extraction method metadata.

```python
def test_extract_preserves_table_and_section(fake_backend, tmp_path):
    result = PdfExtractor(fake_backend).extract(tmp_path / "guide.pdf", "sha256:abc")
    table = next(b for b in result.blocks if b.content_type is ContentType.TABLE)
    assert table.section_path == ("Installation",)
    assert table.page_start == 2
    assert "| Port | Purpose |" in table.text
```

- [ ] **Step 2: Run test and verify failure**

Run: `.venv/bin/pytest tests/test_pdf_extractor.py -v`

Expected: import fails because `pdf_extractor` does not exist.

- [ ] **Step 3: Implement backend protocol and Docling adapter**

Keep Docling imports inside `DoclingBackend` so fake-backed unit tests remain lightweight. Configure local OCR and table-structure recognition. Convert Docling items into normalized blocks in reading order, carrying page and bounding-box provenance. Generate stable block IDs from document ID, content type, page range, and ordinal.

Raise `PdfExtractionError(stage, message)` with stages `convert`, `normalize`, or `empty_document`. Do not swallow per-document failures.

- [ ] **Step 4: Run focused tests and one real-PDF smoke extraction**

Run: `.venv/bin/pytest tests/test_pdf_extractor.py -v`

Run: `.venv/bin/python -c "from pathlib import Path; from pdf_extractor import PdfExtractor; print(PdfExtractor.local().extract(Path('docs/ITS-OB4_Quick_Start_Guide.pdf'), 'smoke').stats)"`

Expected: unit tests pass and smoke output reports four pages with nonzero text blocks.

- [ ] **Step 5: Commit**

```bash
git add pdf_extractor.py tests/test_pdf_extractor.py tests/fixtures/docling_document.json
git commit -m "feat: extract structured PDF content locally"
```

### Task 4: Selective local vision enrichment

**Files:**
- Create: `vision_enrichment.py`
- Create: `tests/test_vision_enrichment.py`
- Create: `tests/fixtures/vision_response.json`

**Interfaces:**
- Consumes: `NormalizedDocument`, `ContentBlock`, local Ollama configuration.
- Produces: `VisionClient.describe(page_png: bytes, nearby_text: str) -> VisionDescription`, `should_enrich(page) -> bool`, and `VisionEnricher.enrich(document, pdf_path) -> NormalizedDocument`.

- [ ] **Step 1: Write failing policy and client tests**

Assert that low-text/image pages are selected, dense text-only pages are skipped, only loopback Ollama URLs are accepted, structured JSON is validated, and figure blocks retain original page provenance.

```python
def test_remote_ollama_url_is_rejected():
    with pytest.raises(ValueError, match="local"):
        VisionClient("https://example.com", "model")

def test_enrichment_adds_page_grounded_figure(fake_client, renderer, document):
    enriched = VisionEnricher(fake_client, renderer).enrich(document, Path("guide.pdf"))
    figure = next(b for b in enriched.blocks if b.content_type is ContentType.FIGURE)
    assert figure.page_start == 3
    assert figure.extraction_method == "ollama_vision"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/pytest tests/test_vision_enrichment.py -v`

Expected: import fails because `vision_enrichment` does not exist.

- [ ] **Step 3: Implement renderer, local client, schema, and selection policy**

Render candidate pages to PNG with PyMuPDF. POST to `${OLLAMA_BASE_URL}/api/chat` with `stream: false`, the base64 page image, nearby extracted text, and a JSON-only schema for `summary`, `visible_text`, `relationships`, `steps`, and `confidence`. Reject non-loopback hosts unless an explicit future configuration permits them; this plan does not add such an override.

Skip enrichment failures per page while recording warnings on `NormalizedDocument`; never replace extracted native text with vision output.

- [ ] **Step 4: Run tests and optional local model smoke test**

Run: `.venv/bin/pytest tests/test_vision_enrichment.py -v`

Run: `ollama show qwen2.5vl:7b`

Expected: tests pass; smoke check either confirms the configured local model or produces a clear setup action without affecting unit tests.

- [ ] **Step 5: Commit**

```bash
git add vision_enrichment.py tests/test_vision_enrichment.py tests/fixtures/vision_response.json
git commit -m "feat: enrich PDF visuals with local Ollama"
```

### Task 5: Structure-aware chunking

**Files:**
- Create: `document_chunker.py`
- Create: `tests/test_document_chunker.py`

**Interfaces:**
- Consumes: `NormalizedDocument` and typed blocks.
- Produces: `DocumentChunker.build(document: NormalizedDocument) -> list[IndexChunk]`.

- [ ] **Step 1: Write failing semantic-boundary tests**

Assert that small tables remain intact, walkthrough steps remain together, figure captions are adjacent to descriptions, oversized blocks split with overlap, and chunk IDs are stable across identical runs.

```python
def test_table_is_not_split_when_under_limit(document_with_table):
    chunks = DocumentChunker(max_chars=1000, overlap_chars=100).build(document_with_table)
    table_chunks = [c for c in chunks if c.content_type is ContentType.TABLE]
    assert len(table_chunks) == 1
    assert table_chunks[0].text.startswith("[TABLE]")

def test_chunk_ids_are_deterministic(document_with_table):
    chunker = DocumentChunker(1000, 100)
    assert [c.chunk_id for c in chunker.build(document_with_table)] == [
        c.chunk_id for c in chunker.build(document_with_table)
    ]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/pytest tests/test_document_chunker.py -v`

Expected: import fails because `document_chunker` does not exist.

- [ ] **Step 3: Implement deterministic semantic chunking**

Group adjacent compatible blocks only within the same document, section, and page neighborhood. Prefix rendered text with the content label and section path. Hash document ID, ordered block IDs, and split ordinal for `chunk_id`. Store `page_start`, `page_end`, `content_type`, `section_path`, `extraction_methods`, and block IDs in metadata.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_document_chunker.py -v`

Expected: all chunking tests pass.

- [ ] **Step 5: Commit**

```bash
git add document_chunker.py tests/test_document_chunker.py
git commit -m "feat: add structure-aware document chunking"
```

### Task 6: Incremental and versioned Chroma index manager

**Files:**
- Create: `index_manager.py`
- Create: `tests/test_index_manager.py`

**Interfaces:**
- Consumes: `IndexChunk`, registry index versions, `OllamaEmbeddings`.
- Produces: `IndexManager.upsert_document(chunks)`, `delete_document(document_id)`, `rebuild(documents)`, `active_db_path()`, and `verify(expected_document_ids, expected_chunk_count)`.

- [ ] **Step 1: Write failing tests against an injectable fake vector store**

Cover stable ID upsert, retry idempotency, metadata-based deletion, failed rebuild preserving active pointer, successful verified rebuild switching pointer, and old-index retirement only after success.

```python
def test_failed_rebuild_keeps_active_pointer(manager, fake_store):
    old = manager.active_db_path()
    fake_store.fail_on_add = True
    with pytest.raises(IndexBuildError):
        manager.rebuild([sample_document])
    assert manager.active_db_path() == old

def test_retry_does_not_duplicate_chunks(manager, chunks):
    manager.upsert_document(chunks)
    manager.upsert_document(chunks)
    assert manager.count(document_id=chunks[0].document_id) == len(chunks)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/pytest tests/test_index_manager.py -v`

Expected: import fails because `index_manager` does not exist.

- [ ] **Step 3: Implement store protocol and Chroma adapter**

Convert `IndexChunk` objects to LangChain `Document`s and pass explicit chunk IDs. Delete any existing IDs for the document inside the new operation before adding deterministic replacements. Write active-index JSON through a temporary file and `os.replace`. Version directories as `db/indexes/<uuid>/`; never build into the active directory.

Verification compares expected chunk count and the set of active document IDs. On failure, close and remove only the temporary version. Do not modify the previous pointer.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_index_manager.py -v`

Expected: all index manager tests pass.

- [ ] **Step 5: Commit**

```bash
git add index_manager.py tests/test_index_manager.py
git commit -m "feat: add recoverable Chroma index management"
```

### Task 7: Compose ingestion CLI and active retrieval

**Files:**
- Modify: `ingest.py`
- Modify: `rag.py`
- Modify: `tools.py`
- Modify: `agent.py`
- Create: `tests/test_ingestion_pipeline.py`
- Create: `tests/test_enriched_retrieval.py`

**Interfaces:**
- Consumes: store, extractor, enricher, chunker, registry, and index manager from Tasks 1–6.
- Produces: `build_ingestion_pipeline()`, `ingest_pdf(path, uploader=None)`, `rebuild_corpus()`, and enriched response source records.

- [ ] **Step 1: Write failing orchestration and retrieval tests**

Use fakes to assert stage order, registry state transitions, cleanup on failure, rebuilding all discovered root/managed PDFs, active-index loading, and formatted context labels/page ranges.

```python
def test_context_preserves_table_label_and_page_range():
    doc = Document(page_content="[TABLE]\n|A|B|", metadata={
        "source": "guide.pdf", "page_start": 4, "page_end": 5,
        "content_type": "table",
    })
    assert "guide.pdf, pages 4-5" in format_context([doc])
    assert "[TABLE]" in format_context([doc])

def test_failed_ingestion_marks_job_failed_and_removes_partial_chunks(pipeline):
    pipeline.index.fail = True
    with pytest.raises(IngestionError):
        pipeline.ingest_pdf(Path("guide.pdf"))
    assert pipeline.registry.latest_job.state == "failed"
    assert pipeline.index.count(document_id="sha256:abc") == 0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/pytest tests/test_ingestion_pipeline.py tests/test_enriched_retrieval.py -v`

Expected: tests fail because the orchestration and enriched metadata are absent.

- [ ] **Step 3: Replace the legacy ingestion composition**

Keep `--reset` as a full versioned rebuild. Add `--add PATH` for local incremental ingestion and `--no-vision` for deterministic operational fallback. Report per-document pages, tables, figures, OCR blocks, chunks, warnings, elapsed time, and final index version.

Update `get_vector_db()` to resolve the active index pointer and fail clearly if no verified index exists. Update source formatting to use one-based `page_start/page_end` while retaining backward compatibility with legacy zero-based `page` metadata during migration. Preserve `run_agent(question)` as the UI/Slack contract.

- [ ] **Step 4: Run focused and legacy tests**

Run: `.venv/bin/pytest tests/test_ingestion_pipeline.py tests/test_enriched_retrieval.py -v`

Run: `.venv/bin/python -m compileall -q agent.py ingest.py rag.py tools.py document_models.py document_store.py pdf_extractor.py vision_enrichment.py document_chunker.py index_manager.py`

Expected: tests pass and compilation exits zero.

- [ ] **Step 5: Commit**

```bash
git add ingest.py rag.py tools.py agent.py tests/test_ingestion_pipeline.py tests/test_enriched_retrieval.py
git commit -m "feat: integrate multimodal ingestion and retrieval"
```

### Task 8: Documentation, complete verification, and corpus acceptance

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Create: `.env.example`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: complete multimodal indexing pipeline.
- Produces: operator setup, privacy constraints, model setup, storage layout, recovery instructions, and verified corpus index.

- [ ] **Step 1: Write configuration regression tests**

Assert staging is excluded, managed PDFs are discoverable, Ollama URL defaults to loopback, and secrets/model settings are not hard-coded into logs.

- [ ] **Step 2: Update operator documentation and ignore rules**

Document `uv sync`, Docling local models, local Ollama vision model pull, environment variables, `--reset`, `--add`, storage layout, GPU expectations, and troubleshooting. Ignore `docs/managed/*`, `docs/staging/*`, registry files, versioned indexes, backups, and local environment files while retaining directory `.gitkeep` files.

- [ ] **Step 3: Run the entire automated suite**

Run: `.venv/bin/pytest -v`

Expected: all tests pass with no network dependency.

- [ ] **Step 4: Rebuild and verify the real corpus**

Run: `.venv/bin/python ingest.py --reset`

Expected: all four current PDFs are registered and the verified active index reports nonzero text/table or figure chunks with source-page metadata. Save only aggregate extraction statistics to logs; do not save extracted content.

- [ ] **Step 5: Exercise representative retrieval**

Ask at least one native-text question, one table question, one image/walkthrough question, and one unsupported question through `run_agent`. Verify correct filenames/pages for supported answers and a grounded refusal for the unsupported question.

- [ ] **Step 6: Commit**

```bash
git add README.md .gitignore .env.example tests/test_config.py
git commit -m "docs: document local multimodal ingestion"
```
