FROM python:3.11-slim

LABEL description="HaemoRL — organ & blood allocation platform with an RL environment"

ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as a non-root user
RUN useradd -m -u 1000 haemorl
WORKDIR /app

# Install dependencies first (Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=haemorl:haemorl . .
USER haemorl

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()" || exit 1

EXPOSE 7860

# One worker: the database lives in process memory, so extra workers would each
# hold a separate copy and requests would see inconsistent state.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--log-level", "info"]
