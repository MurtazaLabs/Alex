"""Integration test: Alembic migrations run cleanly on a fresh database."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

DB_PATH = Path(__file__).parent / "test_migrations.db"


@pytest.fixture
def alembic_env(tmp_path, monkeypatch):
    """Point DATABASE_URL at a temporary SQLite file and run migrations."""
    db_file = tmp_path / "migration_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file.as_posix()}")
    cfg = Config(str(Path(__file__).resolve().parent.parent.parent / "alembic.ini"))
    return cfg


def test_upgrade_head_creates_all_tables(alembic_env):
    command.upgrade(alembic_env, "head")

    # Verify tables were created by inspecting alembic's own version table
    cfg = alembic_env
    import sqlite3

    url = cfg.get_main_option("sqlalchemy.url")
    db_path = url.replace("sqlite+aiosqlite:///", "")
    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()

    assert "alembic_version" in tables
    assert "pairs" in tables
    assert "market_data" in tables
    assert "signals" in tables
    assert "events" in tables
    assert "paper_trades" in tables


def test_downgrade_drops_all_tables(alembic_env):
    command.upgrade(alembic_env, "head")
    command.downgrade(alembic_env, "base")

    import sqlite3

    url = alembic_env.get_main_option("sqlalchemy.url")
    db_path = url.replace("sqlite+aiosqlite:///", "")
    conn = sqlite3.connect(db_path)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert "pairs" not in tables
