# Slack PDF Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let members of one configured Slack management channel safely add, list, and confirm removal of PDFs while reusing the local multimodal ingestion and recoverable index interfaces.

**Architecture:** Focused Slack adapters parse commands/events, authorize by channel, download private PDF files into staging, and delegate lifecycle work to a framework-independent `DocumentLifecycleService`. Removal uses requester-bound expiring confirmations and the existing versioned rebuild mechanism.

**Tech Stack:** Python 3.11, pytest, Slack Bolt/Slack SDK, SQLite registry, concurrent futures, local multimodal ingestion interfaces from the preceding plan.

## Global Constraints

- Management operations are allowed only when the event originates in `RAG_MANAGEMENT_CHANNEL_ID`; all members of that channel are allowed.
- Support explicit `/rag-add`, `/rag-list`, `/rag-remove exact-filename.pdf` commands and natural-language add/remove requests.
- Removal requires confirmation by the original requester in the original channel and deletes both the managed PDF and its indexed content.
- Slack requests acknowledge immediately; document processing does not block the acknowledgement path.
- Private Slack download URLs, document contents, and extracted text must never be logged.
- Existing `/ask-sdk` and app-mention question answering must continue working.

---

## File Structure

- `slack_models.py`: typed Slack management requests, attachments, confirmations, and receipts.
- `slack_document_commands.py`: authorization, parsing, Slack file metadata/download adapter, confirmation UI/state.
- `document_lifecycle.py`: framework-independent add/list/remove orchestration and rollback.
- `slack_bot.py`: thin Bolt event/command/action registration and background dispatch.
- `tests/test_slack_document_commands.py`: parser, authorization, downloader, and confirmation tests.
- `tests/test_document_lifecycle.py`: transactional lifecycle tests with fakes.
- `tests/test_slack_bot.py`: handler acknowledgement and regression tests.

### Task 1: Slack management contracts, authorization, and parsing

**Files:**
- Modify: `config.py`
- Create: `slack_models.py`
- Create: `slack_document_commands.py`
- Create: `tests/test_slack_document_commands.py`

**Interfaces:**
- Produces: `ManagementAction`, `SlackAttachment`, `ManagementRequest`, `ConfirmationRecord`, `authorize_management_channel(channel_id)`, and `parse_management_request(text, files)`.

- [ ] **Step 1: Write failing authorization and parser tests**

Cover missing configuration, allowed/denied channels, `/rag-add`, `/rag-list`, quoted removal filenames, attachment-plus-natural-language add, exact registered filename removal, unrelated messages, and ambiguous multiple attachments.

```python
def test_management_is_authorized_by_channel_not_user(monkeypatch):
    monkeypatch.setattr(config, "RAG_MANAGEMENT_CHANNEL_ID", "C-MANAGE")
    assert authorize_management_channel("C-MANAGE") is True
    assert authorize_management_channel("C-OTHER") is False

def test_attachment_and_add_phrase_is_parsed():
    request = parse_management_request("please add this PDF", [pdf_attachment])
    assert request.action is ManagementAction.ADD
    assert request.attachments == (pdf_attachment,)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/pytest tests/test_slack_document_commands.py -v`

Expected: imports fail because Slack management modules do not exist.

- [ ] **Step 3: Implement contracts and deterministic parser**

Add `RAG_MANAGEMENT_CHANNEL_ID`, `REMOVAL_CONFIRMATION_TTL_SECONDS=300`, and `INGESTION_WORKERS=1` to environment-backed configuration. Normalize Slack message whitespace but preserve filenames. Explicit commands take precedence over natural-language detection. Natural language must contain an add/remove verb and either one PDF attachment or an exact filename; do not route vague conversational messages to destructive operations.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_slack_document_commands.py -v`

Expected: parser and authorization tests pass.

- [ ] **Step 5: Commit**

```bash
git add config.py slack_models.py slack_document_commands.py tests/test_slack_document_commands.py
git commit -m "feat: parse authorized Slack document requests"
```

### Task 2: Authenticated Slack PDF download and validation handoff

**Files:**
- Modify: `slack_document_commands.py`
- Modify: `tests/test_slack_document_commands.py`

**Interfaces:**
- Consumes: `SlackAttachment`, `DocumentStore.stage_bytes`, Slack bot token.
- Produces: `SlackFileDownloader.download(attachment) -> Path` and `SlackDownloadError`.

- [ ] **Step 1: Write failing downloader tests**

Mock the HTTP transport. Assert bearer authentication, redirect handling without leaking authorization to a different host, streamed byte limit, timeout, non-PDF rejection through `DocumentStore`, cleanup after failure, and redacted exceptions.

```python
def test_download_uses_bearer_token_and_stages_pdf(downloader, transport):
    path = downloader.download(pdf_attachment)
    assert transport.last_headers["Authorization"] == "Bearer xoxb-test"
    assert path.parent.name == "staging"

def test_error_never_contains_private_url(downloader, transport):
    transport.raise_timeout = True
    with pytest.raises(SlackDownloadError) as exc:
        downloader.download(pdf_attachment)
    assert "url_private" not in str(exc.value)
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `.venv/bin/pytest tests/test_slack_document_commands.py -k download -v`

Expected: tests fail because `SlackFileDownloader` is absent.

- [ ] **Step 3: Implement injectable streaming downloader**

Accept only Slack-hosted HTTPS private URLs. Send the bot token in an Authorization header, use finite connect/read timeouts, enforce `MAX_PDF_BYTES` while streaming, and pass the result to `DocumentStore.stage_bytes`. Expose sanitized error categories `metadata`, `authorization`, `timeout`, `size`, `content`, and `storage`.

- [ ] **Step 4: Run downloader and store tests**

Run: `.venv/bin/pytest tests/test_slack_document_commands.py tests/test_document_store.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add slack_document_commands.py tests/test_slack_document_commands.py
git commit -m "feat: download Slack PDFs into safe staging"
```

### Task 3: Framework-independent add and list lifecycle service

**Files:**
- Create: `document_lifecycle.py`
- Create: `tests/test_document_lifecycle.py`

**Interfaces:**
- Consumes: downloader output, `DocumentStore`, extraction pipeline, index manager, registry.
- Produces: `DocumentLifecycleService.add(request) -> OperationReceipt`, `list_documents() -> list[DocumentRecord]`, and `OperationError(job_id, stage, message)`.

- [ ] **Step 1: Write failing transactional add/list tests**

Test successful stage/extract/index/promote/register order; content duplicate; filename collision; extraction failure cleanup; index failure cleanup; promotion failure chunk rollback; busy ingestion lock; and list ordering.

```python
def test_add_promotes_only_after_verified_index(service, fakes):
    receipt = service.add(add_request)
    assert fakes.calls == ["validate", "extract", "enrich", "chunk", "index", "verify", "promote", "register"]
    assert receipt.status == "active"

def test_index_failure_leaves_no_active_file_or_chunks(service, fakes):
    fakes.index.raise_on_upsert = True
    with pytest.raises(OperationError) as exc:
        service.add(add_request)
    assert exc.value.stage == "index"
    assert not fakes.store.managed_exists("guide.pdf")
    assert fakes.index.count("sha256:abc") == 0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/pytest tests/test_document_lifecycle.py -v`

Expected: import fails because `document_lifecycle` does not exist.

- [ ] **Step 3: Implement service with one ingestion lock**

Generate a UUID job ID, create registry job state, acquire a nonblocking process/file lock, and transition through `pending`, `extracting`, `indexing`, and `active`. Convert internal exceptions into safe stage-specific `OperationError`s. A receipt includes filename, pages, tables, figures, OCR blocks, chunks, warnings, elapsed milliseconds, and job ID.

- [ ] **Step 4: Run lifecycle tests**

Run: `.venv/bin/pytest tests/test_document_lifecycle.py -v`

Expected: add/list transaction tests pass.

- [ ] **Step 5: Commit**

```bash
git add document_lifecycle.py tests/test_document_lifecycle.py
git commit -m "feat: orchestrate Slack PDF additions"
```

### Task 4: Requester-bound removal confirmation

**Files:**
- Modify: `slack_document_commands.py`
- Modify: `document_lifecycle.py`
- Modify: `tests/test_slack_document_commands.py`
- Modify: `tests/test_document_lifecycle.py`

**Interfaces:**
- Produces: `ConfirmationStore.create(document, requester, channel)`, `consume(token, requester, channel)`, `cancel(token, requester, channel)`, `confirmation_blocks(record)`, and `DocumentLifecycleService.remove_confirmed(record) -> OperationReceipt`.

- [ ] **Step 1: Write failing confirmation and removal tests**

Cover random opaque token storage, no document path in button values, TTL expiration, wrong user, wrong channel, one-time consumption, cancellation, backup-before-delete, rebuild success, rebuild failure restoring file/index, and root bundled PDF removal rejection.

```python
def test_only_original_requester_can_consume_confirmation(store, record):
    token = store.create(record, requester="U1", channel="C1").token
    with pytest.raises(ConfirmationDenied):
        store.consume(token, requester="U2", channel="C1")

def test_failed_rebuild_restores_managed_pdf(service, fakes):
    fakes.index.raise_on_rebuild = True
    with pytest.raises(OperationError):
        service.remove_confirmed(confirmation)
    assert fakes.store.managed_exists("guide.pdf")
    assert fakes.index.active_version == "old"
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `.venv/bin/pytest tests/test_slack_document_commands.py tests/test_document_lifecycle.py -k 'confirm or remove or rebuild' -v`

Expected: tests fail because confirmation/removal interfaces are absent.

- [ ] **Step 3: Implement SQLite confirmation records and recoverable removal**

Store token hash, requester, channel, document ID, created/expiry times, and status. Slack button values contain only the opaque token and action. Removal moves only `docs/managed/` PDFs to a backup, rebuilds from remaining corpus, verifies/switches, marks the record deleted, then permanently removes the backup. Existing bundled `docs/*.pdf` files are listable/queryable but cannot be removed through Slack.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_slack_document_commands.py tests/test_document_lifecycle.py -v`

Expected: all confirmation and lifecycle tests pass.

- [ ] **Step 5: Commit**

```bash
git add slack_document_commands.py document_lifecycle.py tests/test_slack_document_commands.py tests/test_document_lifecycle.py
git commit -m "feat: confirm and safely remove Slack PDFs"
```

### Task 5: Thin Slack Bolt handlers and background execution

**Files:**
- Modify: `slack_bot.py`
- Create: `tests/test_slack_bot.py`

**Interfaces:**
- Consumes: request parser, downloader, lifecycle service, confirmation UI.
- Produces: `/rag-add`, `/rag-list`, `/rag-remove`, document-message event, `rag_confirm_remove`, and `rag_cancel_remove` handlers.

- [ ] **Step 1: Write failing handler tests with a fake Bolt app/service**

Assert acknowledgement occurs before work dispatch, denied channels receive ephemeral responses, progress uses the originating channel/thread, list output is bounded/paginated, actions validate requester/channel, existing `/ask-sdk` behavior remains, and import does not fail when Slack environment variables are absent.

```python
def test_add_acknowledges_before_background_work(handlers, calls):
    handlers.handle_add(ack=lambda: calls.append("ack"), command=add_command,
                        respond=lambda **kw: calls.append("respond"))
    assert calls[0] == "ack"
    assert handlers.executor.submitted == 1

def test_import_does_not_require_tokens(monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    importlib.reload(slack_bot)
    assert slack_bot.create_app is not None
```

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/pytest tests/test_slack_bot.py -v`

Expected: tests fail because handlers are monolithic/import-time configuration raises.

- [ ] **Step 3: Refactor app creation and register management handlers**

Move token validation into `create_app()`/`main()`. Build dependencies there and use a single-worker `ThreadPoolExecutor`. Register explicit commands, action IDs, and a constrained message listener that ignores bot messages and only parses management requests in the management channel. Keep `/ask-sdk` and `app_mention` question behavior intact; prevent a management phrase from also being treated as a question.

Post progress stages without document content. Format final receipts and safe errors with job IDs. Shut down the executor gracefully when Socket Mode exits.

- [ ] **Step 4: Run Slack and regression tests**

Run: `.venv/bin/pytest tests/test_slack_bot.py tests/test_slack_document_commands.py tests/test_document_lifecycle.py -v`

Run: `.venv/bin/python -m compileall -q slack_bot.py slack_models.py slack_document_commands.py document_lifecycle.py`

Expected: all tests pass and compilation exits zero.

- [ ] **Step 5: Commit**

```bash
git add slack_bot.py tests/test_slack_bot.py
git commit -m "feat: expose Slack PDF lifecycle commands"
```

### Task 6: Slack setup documentation and end-to-end acceptance

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Create: `tests/test_slack_lifecycle_acceptance.py`

**Interfaces:**
- Consumes: complete Slack lifecycle and multimodal pipeline.
- Produces: reproducible Slack app scopes/events/commands setup and verified add/list/cancel/remove behavior.

- [ ] **Step 1: Write an acceptance test using real temporary files and fake extraction/index adapters**

The test must add a fixture PDF, list it, reject a duplicate, request removal, cancel removal, confirm it remains, request again, confirm removal, and verify both the managed file and document chunks disappear.

- [ ] **Step 2: Document Slack configuration exactly**

Document required bot scopes (`commands`, `app_mentions:read`, `chat:write`, `files:read`, plus `channels:history` for a public management channel or `groups:history` for a private management channel), Socket Mode, slash commands, interactivity action endpoints handled through Socket Mode, subscribed message/app mention events, management channel ID, confirmation TTL, and restart instructions.

Document that every channel member may manage PDFs only inside the configured channel, bundled root PDFs cannot be removed through Slack, and all AI/OCR processing stays local after download.

- [ ] **Step 3: Run the complete automated suite**

Run: `.venv/bin/pytest -v`

Expected: all multimodal, lifecycle, Slack, Streamlit regression, and acceptance tests pass.

- [ ] **Step 4: Perform live Slack acceptance**

In the configured management channel:

1. Upload a small PDF and invoke `/rag-add` or mention the bot with “please add this PDF.”
2. Verify immediate acknowledgement, progress, final extraction receipt, and `/rag-list` entry.
3. Ask a question whose answer comes from a table or screenshot and verify filename/page citations.
4. Run `/rag-remove exact-filename.pdf`, cancel, and verify it remains.
5. Repeat removal, confirm as the requesting user, and verify the local file and chunks are gone after rebuild.
6. Attempt management from another channel and verify denial.

- [ ] **Step 5: Commit**

```bash
git add README.md .env.example tests/test_slack_lifecycle_acceptance.py
git commit -m "docs: document Slack PDF lifecycle"
```
