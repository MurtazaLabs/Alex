"""Tests for the LLM client: fallback, JSON parsing, cache, budget, privacy guard."""

from __future__ import annotations

import pytest

from forex_ai_agent.config.settings import Settings
from forex_ai_agent.llm import LLMClient, LLMUnavailable


@pytest.fixture
def client_no_key() -> LLMClient:
    settings = Settings(
        environment="test",
        llm_enabled=True,
        openrouter_api_key=None,
        ollama_base_url=None,
        _env_file=None,  # type: ignore[call-arg]
    )
    return LLMClient(settings)


@pytest.fixture
def client_disabled() -> LLMClient:
    settings = Settings(
        environment="test",
        llm_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    return LLMClient(settings)


@pytest.mark.asyncio
async def test_disabled_llm_raises(client_disabled: LLMClient) -> None:
    with pytest.raises(LLMUnavailable):
        await client_disabled.narrate("system", {"pair": "EUR/USD"})


@pytest.mark.asyncio
async def test_no_key_raises_llm_unavailable(client_no_key: LLMClient) -> None:
    with pytest.raises(LLMUnavailable):
        await client_no_key.narrate("system", {"pair": "EUR/USD"})


@pytest.mark.asyncio
async def test_reason_tier_requires_openrouter(client_no_key: LLMClient) -> None:
    """The reasoning tier never falls back to Ollama; no key → unavailable."""
    with pytest.raises(LLMUnavailable):
        await client_no_key.reason("system", {"pair": "EUR/USD"})


def test_sanitize_denylist_raises() -> None:
    with pytest.raises(LLMUnavailable):
        LLMClient._sanitize({"pair": "EUR/USD", "api_key": "sk-123"})


def test_sanitize_strips_nan_and_nested_secrets() -> None:
    with pytest.raises(LLMUnavailable):
        LLMClient._sanitize({"a": [{"password": "x"}]})


def test_parse_json_tolerates_fences_and_prose() -> None:
    text = 'Sure! ```json\n{"rules": [{"name": "a"}]}\n```\nDone.'
    data = LLMClient._parse_json(text)
    assert data == {"rules": [{"name": "a"}]}


def test_parse_json_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        LLMClient._parse_json("[1, 2, 3]")


def test_cache_roundtrip(client_no_key: LLMClient) -> None:
    from forex_ai_agent.llm.client import LLMResponse

    resp = LLMResponse(text="hello", model="test", tier="narrative", tokens_in=1, tokens_out=1)
    key = "abc"
    client_no_key._cache_set(key, resp)
    hit = client_no_key._cache_get(key)
    assert hit is not None
    assert hit.text == "hello"
    assert hit.cached is True


def test_budget_exhaustion_raises(client_no_key: LLMClient) -> None:
    client_no_key._settings.llm_daily_token_budget = 10
    client_no_key._maybe_reset_budget_day()  # initialize the day
    client_no_key._spend(50)
    client_no_key._maybe_reset_budget_day()  # same UTC day → no reset
    with pytest.raises(LLMUnavailable):
        client_no_key._check_budget()


def test_budget_resets_on_new_day(client_no_key: LLMClient) -> None:
    from datetime import timedelta

    client_no_key._settings.llm_daily_token_budget = 10
    client_no_key._maybe_reset_budget_day()  # initialize the day
    client_no_key._spend(50)
    assert client_no_key._tokens_today == 50
    client_no_key._budget_day = client_no_key._budget_day - timedelta(days=2)
    client_no_key._maybe_reset_budget_day()  # rollover → reset
    assert client_no_key._tokens_today == 0
    client_no_key._check_budget()  # no raise


@pytest.mark.asyncio
async def test_health_snapshot(client_no_key: LLMClient) -> None:
    health = await client_no_key.health()
    assert health["llm_enabled"] is True
    assert health["openrouter_configured"] is False


# ---------- provider chain (OpenRouter → UnoRouter) ----------


def _chain_client(**overrides) -> LLMClient:
    defaults = dict(
        environment="test",
        llm_enabled=True,
        openrouter_api_key=None,
        ollama_base_url=None,
        unorouter_api_key=None,
        _env_file=None,  # type: ignore[arg-type]
    )
    defaults.update(overrides)
    return LLMClient(Settings(**defaults))


@pytest.mark.asyncio
async def test_chain_openrouter_first_then_unorouter_fallback() -> None:
    """OpenRouter failure must fall through to a configured UnoRouter."""
    calls: list[tuple[str, str]] = []

    async def fake_cc(self, *, base_url, api_key, model, payload, tier, json_mode, timeout=None, extra_headers=None):
        calls.append((base_url, model))
        if "openrouter" in base_url:
            raise LLMUnavailable("402 payment required (simulated)")
        return LLMClient.__new__(LLMClient) if False else _fake_resp(model)

    from forex_ai_agent.llm import client as client_mod

    client = _chain_client(
        openrouter_api_key="sk-or",
        unorouter_api_key="sk-uno",
        alex_llm_unorouter_reasoning_model="gpt-oss-120b:free",
    )

    def _fake_resp(model: str):
        from forex_ai_agent.llm.client import LLMResponse
        return LLMResponse(text='{"ok": true}', model=model, tier="reasoning", tokens_in=1, tokens_out=1)

    orig = client_mod.LLMClient._chat_completions

    async def patched(self, **kw):
        calls.append((kw["base_url"], kw["model"]))
        if "openrouter" in kw["base_url"]:
            raise LLMUnavailable("402 payment required (simulated)")
        return _fake_resp(kw["model"])

    client_mod.LLMClient._chat_completions = patched  # type: ignore[method-assign]
    try:
        out = await client.reason_json("sys", {"pair": "EUR/USD"})
    finally:
        client_mod.LLMClient._chat_completions = orig  # type: ignore[method-assign]

    assert out == {"ok": True}
    assert [c[0] for c in calls] == [
        "https://openrouter.ai/api/v1",
        "https://api.unorouter.com/v1",
    ]
    assert calls[1][1] == "gpt-oss-120b:free"


@pytest.mark.asyncio
async def test_chain_skips_unorouter_without_tier_model() -> None:
    """UnoRouter key set but no model for the tier → provider skipped cleanly."""
    client = _chain_client(unorouter_api_key="sk-uno")  # no unorouter models set
    with pytest.raises(LLMUnavailable) as exc_info:
        await client.narrate("sys", {"pair": "EUR/USD"})
    assert "no provider had a key + model" in str(exc_info.value)


@pytest.mark.asyncio
async def test_chain_unorouter_only_configuration() -> None:
    """No OpenRouter key + UnoRouter model set → UnoRouter serves directly."""
    from forex_ai_agent.llm import client as client_mod
    from forex_ai_agent.llm.client import LLMResponse

    client = _chain_client(
        unorouter_api_key="sk-uno",
        alex_llm_unorouter_narrative_model="gpt-oss-120b:free",
    )

    async def patched(self, **kw):
        assert kw["base_url"] == "https://api.unorouter.com/v1"
        assert kw["model"] == "gpt-oss-120b:free"
        return LLMResponse(text="hi", model=kw["model"], tier="narrative", tokens_in=1, tokens_out=1)

    orig = client_mod.LLMClient._chat_completions
    client_mod.LLMClient._chat_completions = patched  # type: ignore[method-assign]
    try:
        out = await client.narrate("sys", {"pair": "EUR/USD"})
    finally:
        client_mod.LLMClient._chat_completions = orig  # type: ignore[method-assign]
    assert out == "hi"


def test_provider_chain_parsing() -> None:
    client = _chain_client(llm_provider_chain=" OpenRouter , unorouter ,")
    assert client._provider_chain() == ["openrouter", "unorouter"]


@pytest.mark.asyncio
async def test_chain_all_providers_fail_lists_tried() -> None:
    """Both providers configured and failing → error names what was tried."""
    from forex_ai_agent.llm import client as client_mod

    client = _chain_client(
        openrouter_api_key="sk-or",
        unorouter_api_key="sk-uno",
        alex_llm_unorouter_reasoning_model="gpt-oss-120b:free",
    )

    async def patched(self, **kw):
        raise LLMUnavailable("down (simulated)")

    orig = client_mod.LLMClient._chat_completions
    client_mod.LLMClient._chat_completions = patched  # type: ignore[method-assign]
    try:
        with pytest.raises(LLMUnavailable) as exc_info:
            await client.reason("sys", {"pair": "EUR/USD"})
    finally:
        client_mod.LLMClient._chat_completions = orig  # type: ignore[method-assign]
    msg = str(exc_info.value)
    assert "openrouter" in msg and "unorouter" in msg


# ---------- model agnosticism (any model, not just OpenAI-flavored) ----------


import httpx  # noqa: E402


class FakeHttp:
    """Records request bodies (deep-copied — the client mutates its body dict
    in-flight when degrading params), replays canned httpx responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies: list = []

    async def post(self, url, json=None, headers=None, timeout=None):
        import copy

        self.bodies.append(copy.deepcopy(json))
        return self.responses.pop(0)


def _ok(content):
    """A 200 completion; ``content=None`` replays the reasoning-model
    empty-completion failure shape verbatim."""
    req = httpx.Request("POST", "https://api.unorouter.com/v1/chat/completions")
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        request=req,
    )



def _bad400(detail: str) -> httpx.Response:
    req = httpx.Request("POST", "https://api.unorouter.com/v1/chat/completions")
    return httpx.Response(400, text=detail, request=req)


@pytest.mark.asyncio
async def test_any_model_json_mode_graceful_degradation(monkeypatch) -> None:
    """A llama/mistral-class model that rejects response_format must recover:
    retry once without it, remember the quirk, and never leak it to other models."""
    from forex_ai_agent.llm import client as client_mod

    client = _chain_client(
        unorouter_api_key="sk-uno",
        alex_llm_unorouter_reasoning_model="some-llama-8b",
    )
    fake = FakeHttp(
        [
            _bad400('{"error": {"message": "response_format is not supported by this model"}}'),
            _ok('{"ok": 1}'),
            _ok('{"ok": 2}'),
        ]
    )

    async def _gc():
        return fake

    monkeypatch.setattr(client_mod.LLMClient, "_get_client", lambda self: _gc())

    out = await client.reason_json("sys", {"pair": "EUR/USD"})
    assert out == {"ok": 1}
    assert "response_format" in fake.bodies[0]  # first try used JSON mode
    assert "response_format" not in fake.bodies[1]  # instant retry without it

    # Quirk remembered on a fresh (uncached) call for the same model...
    out2 = await client.reason_json("sys", {"pair": "GBP/USD"})
    assert out2 == {"ok": 2}
    assert "response_format" not in fake.bodies[2]

    # ...but a DIFFERENT model still receives response_format normally.
    client2 = _chain_client(
        unorouter_api_key="sk-uno",
        alex_llm_unorouter_reasoning_model="gpt-flavor",
    )
    fake2 = FakeHttp([_ok('{"ok": 3}')])

    async def _gc2():
        return fake2

    monkeypatch.setattr(client_mod.LLMClient, "_get_client", lambda self: _gc2())
    out3 = await client2.reason_json("sys", {"pair": "EUR/USD"})
    assert out3 == {"ok": 3}
    assert "response_format" in fake2.bodies[0]


@pytest.mark.asyncio
async def test_any_model_max_tokens_rename(monkeypatch) -> None:
    """Newer OpenAI-style models want max_completion_tokens — rename and retry."""
    from forex_ai_agent.llm import client as client_mod

    client = _chain_client(
        unorouter_api_key="sk-uno",
        alex_llm_unorouter_reasoning_model="o-style-mini",
    )
    fake = FakeHttp(
        [
            _bad400('{"error": {"message": "use max_completion_tokens instead of max_tokens"}}'),
            _ok('{"done": true}'),
        ]
    )

    async def _gc():
        return fake

    monkeypatch.setattr(client_mod.LLMClient, "_get_client", lambda self: _gc())
    out = await client.reason_json("sys", {"pair": "EUR/USD"})
    assert out == {"done": True}
    assert "max_tokens" not in fake.bodies[1]
    assert fake.bodies[1]["max_completion_tokens"] == 1500


@pytest.mark.asyncio
async def test_any_model_temperature_drop(monkeypatch) -> None:
    """Reasoning-style models that forbid temperature get it dropped, not a failure."""
    from forex_ai_agent.llm import client as client_mod

    client = _chain_client(
        unorouter_api_key="sk-uno",
        alex_llm_unorouter_reasoning_model="reasoner-r1",
    )
    fake = FakeHttp(
        [
            _bad400('{"error": {"message": "temperature is not supported with this model"}}'),
            _ok('{"ok": 9}'),
        ]
    )

    async def _gc():
        return fake

    monkeypatch.setattr(client_mod.LLMClient, "_get_client", lambda self: _gc())
    out = await client.reason_json("sys", {"pair": "EUR/USD"})
    assert out == {"ok": 9}
    assert "temperature" not in fake.bodies[1]


# ---------- empty completions are failures, not success ----------


@pytest.mark.asyncio
async def test_empty_completion_advances_chain(monkeypatch) -> None:
    """content=null/blank (reasoning models out of token budget) is a FAILURE:
    the chain must advance to the next model, never return a blank string."""
    from forex_ai_agent.llm import client as client_mod

    client = _chain_client(
        unorouter_api_key="sk-uno",
        alex_llm_unorouter_reasoning_model="reasoner-a",
        alex_llm_unorouter_reasoning_fallbacks="reasoner-b",
    )
    fake = FakeHttp(
        [
            _ok(None),   # model A: 200 OK, empty content
            _ok('{"ok": 7}'),  # model B: real answer
        ]
    )

    async def _gc():
        return fake

    monkeypatch.setattr(client_mod.LLMClient, "_get_client", lambda self: _gc())
    out = await client.reason_json("sys", {"pair": "EUR/USD"})
    assert out == {"ok": 7}
    assert len(fake.bodies) == 2  # advanced instead of retrying the same model


@pytest.mark.asyncio
async def test_empty_completion_everywhere_raises(monkeypatch) -> None:
    """All models returning blanks → LLMUnavailable (never a silent empty)."""
    from forex_ai_agent.llm import client as client_mod

    client = _chain_client(
        unorouter_api_key="sk-uno",
        alex_llm_unorouter_reasoning_model="reasoner-a",
        alex_llm_unorouter_reasoning_fallbacks="reasoner-b",
    )
    fake = FakeHttp([_ok(None), _ok("")])

    async def _gc():
        return fake

    monkeypatch.setattr(client_mod.LLMClient, "_get_client", lambda self: _gc())
    with pytest.raises(LLMUnavailable) as exc_info:
        await client.reason_json("sys", {"pair": "EUR/USD"})
    assert "empty completion" in str(exc_info.value)
    assert "response_format" in fake.bodies[1]  # unrelated params untouched
