import importlib


def test_slack_bot_imports_without_credentials(monkeypatch):
    for name in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_SIGNING_SECRET"):
        monkeypatch.delenv(name, raising=False)
    module = importlib.import_module("slack_bot")
    assert module.app is None


def test_list_handler_acknowledges_immediately(monkeypatch):
    import slack_bot

    monkeypatch.setattr(slack_bot, "RAG_MANAGEMENT_CHANNEL_ID", "C1")
    handlers = {}

    class FakeApp:
        def command(self, name):
            return lambda function: handlers.setdefault(name, function) or function

        def event(self, name):
            return lambda function: handlers.setdefault(name, function) or function

        def action(self, name):
            return lambda function: handlers.setdefault(name, function) or function

    service = type("Service", (), {"list_documents": lambda self: []})()
    slack_bot.register_handlers(FakeApp(), service)
    calls = []
    handlers["/rag-list"](
        lambda: calls.append("ack"),
        lambda **kwargs: calls.append(kwargs["text"]),
        {"channel_id": "C1", "user_id": "U1"},
    )
    assert calls == ["ack", "No PDFs are currently indexed."]
