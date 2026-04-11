# ── Stage 1: dependency layer (cached aggressively) ──────────────────────────
FROM python:3.11-slim AS deps

WORKDIR /deps

# Install system deps needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin            /usr/local/bin

# Copy entire application source (this fixes the missing server folder bug)
COPY --chown=appuser:appuser . ./

USER appuser

# HuggingFace Spaces exposes port 7860
EXPOSE 7860

# Health-check so HF Space knows the container is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["python", "app.py"]