"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-08
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from forex_ai_agent.data.models import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all Phase 1 tables from the ORM metadata."""
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    """Drop all application tables."""
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)