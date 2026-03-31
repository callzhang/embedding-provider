FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY provider /app/provider

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV SERVICE_NAME=embedding-provider
ENV MODEL_ID=jinaai/jina-embeddings-v5-text-nano
ENV MODEL_ALIAS=
ENV EMBEDDING_TASK=text-matching
ENV DEFAULT_DIMENSIONS=768
ENV MAX_LENGTH=8192
ENV NORMALIZE_EMBEDDINGS=true
ENV DTYPE=bfloat16
ENV TRUST_REMOTE_CODE=true
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "provider.app:app", "--host", "0.0.0.0", "--port", "8000"]
