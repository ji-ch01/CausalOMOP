FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install omcp first
# Full source (not just pyproject.toml/uv.lock) is required here: omcp's
# pyproject.toml declares readme = "README.md", and hatchling's editable
# build fails validation if that file isn't present yet.
COPY omcp/ /app/omcp/
WORKDIR /app/omcp
RUN uv sync --frozen --no-dev

# Install agno_fastomop
COPY agno_fastomop/ /app/agno_fastomop/
WORKDIR /app/agno_fastomop
RUN uv sync --frozen --no-dev

# ── Runtime image ──────────────────────────────────────────────────────────────
FROM python:3.13-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 fastomop && mkdir -p /app/data && chown -R fastomop:fastomop /app

WORKDIR /app

# Copy venvs from builder
COPY --from=builder /app/omcp/.venv  /app/omcp/.venv
COPY --from=builder /app/agno_fastomop/.venv /app/agno_fastomop/.venv

# Copy source code
COPY --chown=fastomop:fastomop omcp/ /app/omcp/
COPY --chown=fastomop:fastomop agno_fastomop/ /app/agno_fastomop/

USER fastomop

# agno_fastomop venv on PATH
ENV PATH="/app/agno_fastomop/.venv/bin:$PATH"
ENV PYTHONPATH="/app/agno_fastomop/src:$PYTHONPATH"

# Tell agno_fastomop where to find the omcp MCP server
ENV MCP_COMMAND="/app/omcp/.venv/bin/python /app/omcp/src/omcp/main.py"
ENV OTEL_SDK_DISABLED="true"

EXPOSE 7777

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7777/health || exit 1

CMD ["python", "-m", "agno_fastomop.web_interface"]
