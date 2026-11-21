FROM python:3.15-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 OMP_NUM_THREADS=1
WORKDIR /app

# dependencies first: this layer is cached until uv.lock changes
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-install-project --no-dev

# then the code the API needs
COPY api/ api/
COPY src/ src/
COPY params.yaml ./
RUN uv sync --locked --no-dev

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]