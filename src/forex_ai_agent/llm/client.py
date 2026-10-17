"""LLM client for Alex.

Two usage tiers:
- ``narrate``  — cheap/fast: signal & explanation narration.
- ``reason``   — strong: research synthesis & strategy reasoning.

Remote provider chain (first provider with a key + a model for the tier wins):

    Ollama (local, narration only) → OpenRouter → UnoRouter

UnoRouter (https://api.unorouter.com/v1) is OpenAI-compatible; a tier routes
through it only when its per-tier model setting is set, and provider order is
configurable via ``llm_provider_chain`` (default ``openrouter,unorouter``).

Hard rules enforced here (Alex's constitution, mechanically):
- The LLM never receives credentials or user-private data (denylist guard raises).
- The LLM never decides trades; callers only ever feed it deterministic context dicts.
- No usable provider, disabled flag, exhausted budget or network failure →
  ``LLMUnavailable``; callers are expected to fall back to deterministic templates.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Dict, Optional

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)

# Keys that must never be serialized into a prompt (privacy invariant).
DENYLIST = frozenset(
    {
        "password",
        "api_key",
        "apikey",
        "secret",
        "secret_key",
        "token",
        "credential",
        "credentials",
        "authorization",
        "user_profile",
        "account_equity",
    }
)

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

# Default per-request ceiling for a single OpenRouter completion. Tier-specific
# timeouts (settings.llm_narrate_timeout_s / llm_reason_timeout_s) override this.
_DEFAULT_REQUEST_TIMEOUT_S = 120.0


class LLMUnavailable(RuntimeError):
    """No LLM backend is reachable, enabled, or within budget."""


@dataclass
class LLMResponse:
    """A single LLM completion with usage metadata."""

    text: str
    model: str
    tier: str  # "narrative" | "reasoning"
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False


class LLMClient:
    """Two-tier LLM client with caching, budgeting and graceful failure."""

    _CACHE_TTL = timedelta(hours=1)
    _OLLAMA_PROBE_INTERVAL_S = 300.0

    # Model-agnosticism: providers gatekeep the same /chat/completions protocol
    # for EVERY model they host (OpenAI, Anthropic, Google, DeepSeek, Qwen,
    # Meta, Mistral...). The one real protocol difference is optional params:
    # some models 400 on ``response_format``, ``temperature`` or ``max_tokens``.
    # When a 400 error body names a param, we remember the (provider, model)
    # pair and never send that param to it again — JSON output is still
    # obtained via the prompt-level schema instruction, so behavior holds.
    _unsupported_params: ClassVar[Dict[tuple, set]] = {}
    _MAX_RETRIES = 3

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings or get_settings()
        self._cache: Dict[str, tuple[LLMResponse, datetime]] = {}
        self._tokens_today = 0
        self._budget_day: Optional[datetime] = None
        self._ollama_available: Optional[bool] = None
        self._ollama_checked_at: float = 0.0
        self._http: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def narrate(self, system: str, context: Dict[str, Any]) -> str:
        """Cheap, fast narration (signal one-liners, explanation intros)."""
        resp = await self._complete(
            "narrative", system, context, timeout=self._settings.llm_narrate_timeout_s
        )
        return resp.text

    async def reason(self, system: str, context: Dict[str, Any]) -> str:
        """Strong-model reasoning (research synthesis, strategy proposals)."""
        resp = await self._complete(
            "reasoning", system, context, timeout=self._settings.llm_reason_timeout_s
        )
        return resp.text

    async def reason_json(
        self, system: str, context: Dict[str, Any], schema_hint: str = ""
    ) -> Dict[str, Any]:
        """Strong-model reasoning constrained to a JSON object response."""
        instruction = system
        if schema_hint:
            instruction += f"\n\nRespond with ONLY a JSON object matching this shape:\n{schema_hint}"
        else:
            instruction += "\n\nRespond with ONLY a valid JSON object."
        resp = await self._complete("reasoning", instruction, context, json_mode=True)
        return self._parse_json(resp.text)

    async def health(self) -> Dict[str, Any]:
        """Status snapshot for the CLI/dashboard status pages."""
        return {
            "llm_enabled": bool(self._settings.llm_enabled),
            "openrouter_configured": bool(self._settings.openrouter_api_key),
            "unorouter_configured": bool(self._settings.unorouter_api_key),
            "provider_chain": self._provider_chain(),
            "narrative_model": self._settings.alex_llm_narrative_model,
            "reasoning_model": self._settings.alex_llm_reasoning_model,
            "ollama_available": self._ollama_ok() if self._settings.ollama_base_url else False,
            "tokens_used_today": self._tokens_today,
            "daily_budget": self._settings.llm_daily_token_budget,
            "cached_prompts": len(self._cache),
        }

    async def aclose(self) -> None:
        """Release the shared HTTP client."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Completion core
    # ------------------------------------------------------------------

    async def _complete(
        self,
        tier: str,
        system: str,
        context: Dict[str, Any],
        json_mode: bool = False,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        if not self._settings.llm_enabled:
            raise LLMUnavailable("LLM disabled via LLM_ENABLED=false")

        payload = self._build_payload(system, context)
        cache_key = hashlib.sha256(
            f"{tier}|{json_mode}|{payload}".encode("utf-8")
        ).hexdigest()

        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Budget applies to network spend; cached calls are always free.
        self._maybe_reset_budget_day()
        self._check_budget()

        # Narrative tier: try local Ollama first (free, private, fast).
        if tier == "narrative" and self._settings.ollama_base_url and self._ollama_ok():
            try:
                resp = await self._ollama_complete(payload)
                self._cache_set(cache_key, resp)
                return resp
            except Exception as exc:  # noqa: BLE001 — fall through to OpenRouter
                logger.warning("Ollama narrate failed (%s); falling back to OpenRouter", exc)

        # Remote chain: first provider with a key AND a model for this tier wins.
        tried: list = []
        for provider in self._provider_chain():
            model = ""
            extra_headers: Optional[Dict[str, str]] = None
            if provider == "openrouter":
                if not self._settings.openrouter_api_key:
                    continue
                base_url = self._settings.openrouter_base_url
                api_key = self._settings.openrouter_api_key or ""
                extra_headers = {
                    "HTTP-Referer": "https://localhost/forex-ai-agent",
                    "X-Title": "Alex Forex Agent",
                }
                models = self._tier_models(
                    tier,
                    primary=(
                        self._settings.alex_llm_reasoning_model
                        if tier == "reasoning"
                        else self._settings.alex_llm_narrative_model
                    ),
                    fallbacks=(
                        self._settings.alex_llm_reasoning_fallbacks
                        if tier == "reasoning"
                        else self._settings.alex_llm_narrative_fallbacks
                    ),
                )
            elif provider == "unorouter":
                if not self._settings.unorouter_api_key:
                    continue
                base_url = self._settings.unorouter_base_url
                api_key = self._settings.unorouter_api_key or ""
                extra_headers = None
                primary = (
                    self._settings.alex_llm_unorouter_reasoning_model
                    if tier == "reasoning"
                    else self._settings.alex_llm_unorouter_narrative_model
                )
                if not primary:
                    continue  # tier not configured for UnoRouter — skip provider
                models = self._tier_models(
                    tier,
                    primary=primary,
                    fallbacks=(
                        self._settings.alex_llm_unorouter_reasoning_fallbacks
                        if tier == "reasoning"
                        else self._settings.alex_llm_unorouter_narrative_fallbacks
                    ),
                )
            else:
                logger.warning("Unknown provider in llm_provider_chain: %r", provider)
                continue
            # Try every model for this tier on this provider before moving on —
            # free models flap independently, so a saturated primary must not
            # take the whole provider (or chain) down.
            provider_errs: list = []
            for model in models:
                try:
                    resp = await self._chat_completions(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        payload=payload,
                        tier=tier,
                        json_mode=json_mode,
                        timeout=timeout,
                        extra_headers=extra_headers,
                    )
                    self._cache_set(cache_key, resp)
                    return resp
                except LLMUnavailable as exc:
                    provider_errs.append(f"{model}: {exc}")
                    logger.warning("LLM model %s failed (%s); trying next model", model, tier)
            tried.append(f"{provider}: " + " | ".join(provider_errs))
        raise LLMUnavailable(
            "No LLM provider succeeded for the '%s' tier — chain: %s%s"
            % (
                tier,
                " → ".join(self._provider_chain()) or "(empty)",
                (" | " + " | ".join(tried)) if tried else " (no provider had a key + model for this tier)",
            )
        )

    @staticmethod
    def _build_payload(system: str, context: Dict[str, Any]) -> str:
        """Serialize system prompt + sanitized context into one prompt string."""
        safe_context = LLMClient._sanitize(context)
        try:
            context_json = json.dumps(safe_context, default=str, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise LLMUnavailable(f"Context is not JSON-serializable: {exc}") from exc
        return f"{system}\n\nCONTEXT (deterministic data — narrate, never decide):\n{context_json}"

    @classmethod
    def _sanitize(cls, obj: Any, path: str = "$") -> Any:
        """Recursively strip restricted keys and JSON-hostile values."""
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for key, value in obj.items():
                if str(key).lower() in DENYLIST:
                    raise LLMUnavailable(
                        f"Refusing to send restricted key {path}.{key} to the LLM"
                    )
                out[str(key)] = cls._sanitize(value, f"{path}.{key}")
            return out
        if isinstance(obj, (list, tuple, set)):
            return [cls._sanitize(v, f"{path}[{i}]") for i, v in enumerate(obj)]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        # numpy scalars → python natives (has .item(), is not a str/dict)
        if hasattr(obj, "item") and not isinstance(obj, (str, bytes, dict, list)):
            return cls._sanitize(obj.item(), path)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse a JSON object out of a model response (any-model tolerant).

        Handles the realistic free-model quirks: code fences, prose before/
        after the object, single-quoted strings, trailing commas, unquoted
        keys, and python-style booleans — repairs when legal, never guesses.
        """
        cleaned = text.strip()
        fence = _CODE_FENCE_RE.match(cleaned)
        if fence:
            cleaned = fence.group(1).strip()
        # Tolerate prose before/after the outermost JSON object.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                # "Extra data" quirk: a valid object followed by commentary —
                # raw_decode takes the FIRST complete object, ignores the rest.
                data, _ = json.JSONDecoder().raw_decode(cleaned)
            except json.JSONDecodeError:
                # Repair pass — only for quirks models actually emit.
                repaired = re.sub(r"\bTrue\b", "true", cleaned)
                repaired = re.sub(r"\bFalse\b", "false", repaired)
                repaired = re.sub(r"\bNone\b", "null", repaired)
                repaired = re.sub(r",\s*([}\]])", r"\1", repaired)  # trailing commas
                repaired = re.sub(
                    r"([{,]\s*)'([^']+)'(\s*:)", r'\1"\2"\3', repaired
                )  # single-quoted keys
                repaired = re.sub(
                    r"(:\s*)'([^']*)'", r'\1"\2"', repaired
                )  # single-quoted values
                repaired = re.sub(
                    r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', repaired
                )  # unquoted keys
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError:
                    data, _ = json.JSONDecoder().raw_decode(repaired)  # repaired + trailing junk
        if not isinstance(data, dict):
            raise ValueError("LLM JSON response is not an object")
        return data

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _ollama_ok(self) -> bool:
        """Cheap cached probe of the local Ollama server."""
        base = (self._settings.ollama_base_url or "").rstrip("/")
        if not base:
            return False
        now = time.monotonic()
        if self._ollama_available is not None and now - self._ollama_checked_at < self._OLLAMA_PROBE_INTERVAL_S:
            return self._ollama_available
        try:
            resp = httpx.get(f"{base}/api/tags", timeout=0.5)
            self._ollama_available = resp.status_code == 200
        except Exception:  # noqa: BLE001 — probe failures are expected
            self._ollama_available = False
        self._ollama_checked_at = now
        return self._ollama_available

    async def _ollama_complete(self, payload: str) -> LLMResponse:
        base = (self._settings.ollama_base_url or "").rstrip("/")
        # Local model name: any narrative model without an "org/name" slash.
        model = self._settings.alex_llm_narrative_model
        if "/" in model:
            model = "llama3.2"
        client = await self._get_client()
        resp = await client.post(
            f"{base}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": payload}],
                "stream": False,
                "options": {"temperature": 0.4},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("message") or {}).get("content", "")
        return LLMResponse(
            text=text,
            model=f"ollama/{model}",
            tier="narrative",
            tokens_in=int(data.get("prompt_eval_count") or 0),
            tokens_out=int(data.get("eval_count") or 0),
        )

    def _provider_chain(self) -> list:
        """Parse ``llm_provider_chain`` into a clean lowercase provider list."""
        return [
            p.strip().lower()
            for p in (self._settings.llm_provider_chain or "").split(",")
            if p.strip()
        ]

    @staticmethod
    def _tier_models(tier: str, primary: str, fallbacks: str) -> list:
        """Primary model + comma-separated fallbacks, deduplicated, primary first."""
        models = [primary.strip()] if primary and primary.strip() else []
        for m in (fallbacks or "").split(","):
            m = m.strip()
            if m and m not in models:
                models.append(m)
        return models

    async def _chat_completions(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        payload: str,
        tier: str,
        json_mode: bool,
        timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        """One OpenAI-compatible ``/chat/completions`` call with retries.

        429 honors ``retry-after``; other errors back off exponentially.
        Raises ``LLMUnavailable`` on final failure so the caller can move to
        the next provider in the chain.
        """
        dropped = self._unsupported_params.get((base_url, model), set())
        body: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": payload}],
            "temperature": 0.2 if tier == "reasoning" else 0.4,
            "max_tokens": 1500 if tier == "reasoning" else 400,
        }
        if json_mode and "response_format" not in dropped:
            body["response_format"] = {"type": "json_object"}
        for param in dropped:
            body.pop(param, None)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        url = f"{base_url.rstrip('/')}/chat/completions"

        last_err: Optional[Exception] = None
        effective_timeout = timeout if timeout is not None else _DEFAULT_REQUEST_TIMEOUT_S
        for attempt in range(self._MAX_RETRIES):
            try:
                client = await self._get_client()
                resp = await client.post(
                    url, json=body, headers=headers, timeout=effective_timeout
                )
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("retry-after") or 0)
                    # Free tiers often omit retry-after and enforce per-minute
                    # TOKEN windows — a big prompt can 429 instantly. Back off
                    # well past the window instead of retrying into the wall.
                    wait = retry_after if retry_after > 0 else 4.0 * (2**attempt)
                    last_err = LLMUnavailable(f"rate limited (429) after {attempt + 1} attempts")
                    logger.warning("%s rate limited; retrying in %.1fs", model, wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = (data["choices"][0]["message"]["content"]) or ""
                if not text.strip():
                    # An empty completion is NOT success: reasoning-style models
                    # burn their token budget on hidden reasoning and return
                    # content=None. Treat it like a model failure so the chain
                    # advances to the next model/provider instead of handing a
                    # blank string to callers (who then see JSONDecodeError).
                    # Raise immediately — no same-model retries: the behavior is
                    # deterministic for this prompt, and a retry costs money.
                    logger.warning("%s returned an empty completion; trying next model", model)
                    raise LLMUnavailable(f"{model} returned an empty completion (content=null/blank)")
                usage = data.get("usage") or {}
                out = LLMResponse(
                    text=text,
                    model=model,
                    tier=tier,
                    tokens_in=int(usage.get("prompt_tokens") or 0),
                    tokens_out=int(usage.get("completion_tokens") or 0),
                )
                self._spend(out.tokens_in + out.tokens_out)
                return out
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                last_err = exc
                # Model-agnostic graceful degradation: a 400 whose body names an
                # optional param means THIS model rejects it. Drop the named
                # param for this (provider, model) pair and retry immediately.
                if status == 400 and exc.response is not None:
                    detail = ""
                    try:
                        detail = (exc.response.text or "").lower()
                    except Exception:  # noqa: BLE001 — body may be unreadable
                        detail = ""
                    named = [
                        p
                        for p in ("response_format", "temperature")
                        if p in body and (p in detail or p.replace("_", " ") in detail)
                    ]
                    if "max_tokens" in body and (
                        "max_tokens" in detail or "max_completion_tokens" in detail
                    ):
                        # Newer OpenAI-style models want the new name.
                        body["max_completion_tokens"] = body.pop("max_tokens")
                        named.append("max_tokens->max_completion_tokens")
                    if named:
                        self._unsupported_params.setdefault((base_url, model), set()).update(
                            p for p in named if "->" not in p
                        )
                        for p in named:
                            if "->" not in p:
                                body.pop(p, None)  # apply now — body is reused on retry
                        logger.info(
                            "Model %s rejected %s; retrying without it "
                            "(prompt-level instructions still enforce behavior)",
                            model,
                            ", ".join(named),
                        )
                        continue
                await asyncio.sleep(0.8 * (2**attempt))
            except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as exc:
                last_err = exc
                await asyncio.sleep(0.8 * (2**attempt))
        raise LLMUnavailable(
            f"{model} via {base_url} failed after {self._MAX_RETRIES} attempts: {last_err}"
        )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(_DEFAULT_REQUEST_TIMEOUT_S, connect=5.0)
            )
        return self._http

    # ------------------------------------------------------------------
    # Cache & budget
    # ------------------------------------------------------------------

    def _cache_get(self, key: str) -> Optional[LLMResponse]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        resp, stored_at = entry
        if datetime.now(timezone.utc) - stored_at > self._CACHE_TTL:
            del self._cache[key]
            return None
        return LLMResponse(
            text=resp.text,
            model=resp.model,
            tier=resp.tier,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cached=True,
        )

    def _cache_set(self, key: str, resp: LLMResponse) -> None:
        self._cache[key] = (resp, datetime.now(timezone.utc))

    def _check_budget(self) -> None:
        """Raise when today's token spend has hit the configured ceiling."""
        if self._tokens_today >= self._settings.llm_daily_token_budget:
            raise LLMUnavailable(
                f"Daily LLM token budget exhausted ({self._tokens_today}/{self._settings.llm_daily_token_budget})"
            )

    def _maybe_reset_budget_day(self) -> None:
        """Lazily adopt/roll the UTC day; only a real rollover resets the counter."""
        now = datetime.now(timezone.utc)
        if self._budget_day is not None and self._budget_day.date() != now.date():
            self._tokens_today = 0
        self._budget_day = now

    def _spend(self, tokens: int) -> None:
        self._tokens_today += max(tokens, 0)
