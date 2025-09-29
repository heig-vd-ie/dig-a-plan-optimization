FROM python:3.12.3-slim

ENV POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    SERVER_PY_PORT=8001 \
    PYTHONPATH=/app/src \
    LOCAL_HOST=0.0.0.0 \
    SERVER_JL_PORT=8081 \
    PYTHONWARNINGS="ignore::SyntaxWarning"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    python3-dev \
    build-essential \
    direnv \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app

COPY pyproject.toml poetry.lock /app/
COPY external-dist/ /app/external-dist/
COPY scripts/ /app/scripts/

RUN poetry install --extras "internal"

COPY src/ /app/src/
COPY examples/ /app/examples/
COPY data/ /app/data/

ENV GRB_LICENCE_FILE=/licenses/GRB_LICENCE_FILE

EXPOSE ${SERVER_PY_PORT}

# 10. Run Python API with uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $SERVER_PY_PORT"]
