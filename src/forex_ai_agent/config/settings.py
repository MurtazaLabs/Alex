"""
Application settings using Pydantic Settings (Pydantic v2).

All values can be overridden by environment variables or a local ``.env`` file.
Environment variable names are derived from field names (case-insensitive),
e.g. ``database_url`` <-> ``DATABASE_URL``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / ``.env`` file."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database (local-first: this default only applies when DATABASE_URL is
    # provided by neither the environment nor .env — every surface (API, CLI,
    # dashboard) then runs on local SQLite. Production sets DATABASE_URL.)
    database_url: str = Field(
        default_factory=lambda: "sqlite+aiosqlite:///alex_local.db"
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379")

    # Environment
    environment: str = Field(default="development")

    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production")

    # API Keys (for external services - live/broker trading requires Level 4)
    alpaca_api_key: Optional[str] = Field(default=None)
    alpaca_secret_key: Optional[str] = Field(default=None)
    mt5_login: Optional[int] = Field(default=None)
    mt5_password: Optional[str] = Field(default=None)
    mt5_server: Optional[str] = Field(default=None)
    mt5_path: Optional[str] = Field(default=None)

    # Trading settings
    default_risk_percent: float = Field(default=0.5, ge=0.1, le=5.0)
    min_risk_reward: float = Field(default=2.0, ge=1.0)
    max_total_exposure_pct: float = Field(default=5.0, ge=0.1, le=20.0)
    max_position_size_lots: float = Field(default=10.0, ge=0.1)

    # Timeframes
    default_timeframes: List[str] = Field(default=["H1", "H4"])

    # Session filtering
    allowed_sessions: List[str] = Field(default=["London", "NewYork"])

    # Permission levels (0=research only ... 4=live trading)
    max_permission_level: int = Field(default=0, ge=0, le=4)

    # News filtering
    news_filter_enabled: bool = Field(default=True)

    # Model settings
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")

    # LLM routing (remote chain: OpenRouter → UnoRouter fallback; local Ollama first for narration)
    openrouter_api_key: Optional[str] = Field(default=None)
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    alex_llm_narrative_model: str = Field(default="openai/gpt-4o-mini")
    alex_llm_reasoning_model: str = Field(default="openai/gpt-4.1-mini")
    # Free models flap independently (429 when saturated upstream, 404 when
    # retired) — comma-separated fallback lists are tried in order per tier.
    alex_llm_narrative_fallbacks: str = Field(default="")
    alex_llm_reasoning_fallbacks: str = Field(default="")
    # UnoRouter (OpenAI-compatible gateway, https://api.unorouter.com/v1).
    # A tier is routed through UnoRouter only when its model below is set —
    # unset means "skip this provider for that tier".
    unorouter_api_key: Optional[str] = Field(default=None)
    unorouter_base_url: str = Field(default="https://api.unorouter.com/v1")
    alex_llm_unorouter_narrative_model: Optional[str] = Field(default=None)
    alex_llm_unorouter_reasoning_model: Optional[str] = Field(default=None)
    alex_llm_unorouter_narrative_fallbacks: str = Field(default="")
    alex_llm_unorouter_reasoning_fallbacks: str = Field(default="")
    # Remote provider order (comma-separated). First provider with a key and a
    # model for the tier wins; later entries are fallbacks.
    llm_provider_chain: str = Field(default="openrouter,unorouter")
    ollama_base_url: Optional[str] = Field(default=None)
    llm_enabled: bool = Field(default=True)
    llm_daily_token_budget: int = Field(default=200_000, ge=1000)
    # Per-call ceilings (seconds). Free OpenRouter models can queue 30-60s;
    # narration waits patiently rather than silently falling back to templates.
    llm_narrate_timeout_s: float = Field(default=90.0, gt=0)
    llm_reason_timeout_s: float = Field(default=120.0, gt=0)

    # Data feed failover chain (first healthy provider wins; comma-separated)
    data_feed_chain: str = Field(default="mt5,yahoo,twelvedata,mock")
    twelvedata_api_key: Optional[str] = Field(default=None)
    feed_health_cooldown_s: int = Field(default=300, ge=10)
    account_equity: float = Field(default=10_000.0, gt=0)

    # Discord notifications (webhook URL; set via DISCORD_WEBHOOK_URL env var)
    discord_webhook_url: Optional[str] = Field(default=None)
    discord_notify_enabled: bool = Field(default=True)
    discord_notify_trades: bool = Field(default=True)
    discord_notify_errors: bool = Field(default=True)
    discord_notify_daily_report: bool = Field(default=True)

    @field_validator("default_timeframes", "allowed_sessions", mode="before")
    @classmethod
    def _parse_json_list(cls, value: object) -> object:
        """Parse JSON-encoded (or comma-separated) lists from env variables."""
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
            return [item.strip() for item in text.split(",") if item.strip()]
        return value

    @property
    def is_production(self) -> bool:
        """Whether the app runs in the production environment."""
        return self.environment.lower() == "production"

    @property
    def is_test(self) -> bool:
        """Whether the app runs in the test environment."""
        return self.environment.lower() == "test"


@lru_cache
def get_settings() -> Settings:
    """Return the application settings singleton."""
    return Settings()


# Backwards-compatible module-level singleton
settings = get_settings()


__all__ = ["Settings", "get_settings", "settings"]
