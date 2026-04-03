from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: str | None = None) -> int | None:
    value = os.getenv(name, default)
    if value in (None, ""):
        return None
    return int(value)


@dataclass(frozen=True)
class Settings:
    service_name: str
    model_id: str
    model_alias: str | None
    api_key: str | None
    embedding_task: str | None
    default_dimensions: int | None
    max_length: int | None
    max_batch_size: int | None
    normalize_embeddings: bool
    dtype: str
    attn_implementation: str | None
    trust_remote_code: bool
    batch_window_ms: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            service_name=os.getenv("SERVICE_NAME", "embedding-provider"),
            model_id=os.getenv("MODEL_ID", "jinaai/jina-embeddings-v5-text-nano"),
            model_alias=os.getenv("MODEL_ALIAS") or None,
            api_key=os.getenv("API_KEY") or None,
            embedding_task=os.getenv("EMBEDDING_TASK") or None,
            default_dimensions=_env_int("DEFAULT_DIMENSIONS", "768"),
            max_length=_env_int("MAX_LENGTH", "8192"),
            max_batch_size=_env_int("MAX_BATCH_SIZE", "8"),
            normalize_embeddings=_env_bool("NORMALIZE_EMBEDDINGS", True),
            dtype=os.getenv("DTYPE", "bfloat16"),
            attn_implementation=os.getenv("ATTN_IMPLEMENTATION") or None,
            trust_remote_code=_env_bool("TRUST_REMOTE_CODE", True),
            batch_window_ms=_env_int("BATCH_WINDOW_MS", "200") or 200,
        )
