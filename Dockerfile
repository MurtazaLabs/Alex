FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for asyncpg / uvicorn
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leverages Docker layer caching)
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8000

# Apply database migrations, then boot the API
CMD ["sh", "-c", "alembic upgrade head && uvicorn forex_ai_agent.main:app --host 0.0.0.0 --port 8000"]