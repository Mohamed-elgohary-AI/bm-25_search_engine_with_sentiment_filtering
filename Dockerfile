# Dockerfile
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN useradd -m -u 1000 appuser
RUN uv sync --frozen --no-dev

COPY . .
RUN chown -R appuser:appuser /app
USER appuser
COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]