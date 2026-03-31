from __future__ import annotations

import inspect
import logging
import math
import time
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModel

from provider.config import Settings

log = logging.getLogger("embedding_provider")
logging.basicConfig(level="INFO")


def _batched(values: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        return [values]
    return [values[idx: idx + batch_size] for idx in range(0, len(values), batch_size)]


def _is_cuda_oom(exc: Exception) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "CUDA out of memory" in str(exc)


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None
    dimensions: int | None = None
    encoding_format: Literal["float"] | None = "float"
    user: str | None = None
    task: str | None = None


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "stardust"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


class EmbedderRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = str(self._device)
        self._model = self._load_model()
        self._encode_signature = inspect.signature(self._model.encode)

    def _resolve_dtype(self) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(self._settings.dtype.lower(), torch.bfloat16)

    def _load_model(self) -> Any:
        kwargs: dict[str, Any] = {
            "trust_remote_code": self._settings.trust_remote_code,
            "torch_dtype": self._resolve_dtype(),
        }
        if self._settings.attn_implementation:
            kwargs["attn_implementation"] = self._settings.attn_implementation
        try:
            model = AutoModel.from_pretrained(self._settings.model_id, **kwargs)
        except Exception:
            log.warning("model load with custom attention failed; retrying without attn_implementation", exc_info=True)
            kwargs.pop("attn_implementation", None)
            model = AutoModel.from_pretrained(self._settings.model_id, **kwargs)
        if hasattr(model, "to"):
            model = model.to(self._device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def _encode_once(self, texts: list[str], *, dimensions: int | None = None, task: str | None = None) -> list[list[float]]:
        kwargs: dict[str, Any] = {}
        input_name = "texts" if "texts" in self._encode_signature.parameters else "sentences"
        requested_dimensions = dimensions or self._settings.default_dimensions
        effective_task = task or self._settings.embedding_task
        if effective_task and "task" in self._encode_signature.parameters:
            kwargs["task"] = effective_task
        if requested_dimensions and "truncate_dim" in self._encode_signature.parameters:
            kwargs["truncate_dim"] = requested_dimensions
        if self._settings.max_length and "max_length" in self._encode_signature.parameters:
            kwargs["max_length"] = self._settings.max_length

        start = time.perf_counter()
        outputs = self._model.encode(**{input_name: texts}, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info("model=%s texts=%s elapsed_ms=%.1f", self._settings.model_id, len(texts), elapsed_ms)

        array = _outputs_to_numpy(outputs)
        if array.ndim == 1:
            array = np.expand_dims(array, axis=0)
        if requested_dimensions and "truncate_dim" not in self._encode_signature.parameters:
            if requested_dimensions > array.shape[1]:
                raise ValueError(f"Requested dimensions={requested_dimensions} but model only produced {array.shape[1]}")
            array = array[:, :requested_dimensions]
        if self._settings.normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            array = array / norms
        return array.tolist()

    def _encode_with_backoff(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        task: str | None = None,
    ) -> list[list[float]]:
        try:
            return self._encode_once(texts, dimensions=dimensions, task=task)
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(texts) <= 1:
                raise ValueError(
                    "Embedding request exceeded GPU memory for a single input; reduce input length or lower MAX_LENGTH"
                ) from exc
            split_at = max(1, len(texts) // 2)
            log.warning(
                "CUDA OOM for model=%s batch_size=%s; retrying with smaller batches",
                self._settings.model_id,
                len(texts),
            )
            left = self._encode_with_backoff(texts[:split_at], dimensions=dimensions, task=task)
            right = self._encode_with_backoff(texts[split_at:], dimensions=dimensions, task=task)
            return [*left, *right]

    def encode(self, texts: list[str], *, dimensions: int | None = None, task: str | None = None) -> list[list[float]]:
        max_batch_size = self._settings.max_batch_size or len(texts)
        embeddings: list[list[float]] = []
        for chunk in _batched(texts, max_batch_size):
            embeddings.extend(self._encode_with_backoff(chunk, dimensions=dimensions, task=task))
        return embeddings


def _outputs_to_numpy(outputs: Any) -> np.ndarray:
    if isinstance(outputs, torch.Tensor):
        return outputs.detach().float().cpu().numpy()
    if isinstance(outputs, np.ndarray):
        return outputs.astype(np.float32, copy=False)
    if isinstance(outputs, list):
        rows: list[np.ndarray] = []
        for item in outputs:
            if isinstance(item, torch.Tensor):
                rows.append(item.detach().float().cpu().numpy())
            else:
                rows.append(np.asarray(item, dtype=np.float32))
        return np.stack(rows, axis=0)
    return np.asarray(outputs, dtype=np.float32)


def _estimate_tokens(texts: list[str]) -> int:
    return sum(max(1, math.ceil(len(text) / 4)) for text in texts)


def _validate_inputs(value: str | list[str]) -> list[str]:
    texts = [value] if isinstance(value, str) else value
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")
    if not all(isinstance(item, str) and item.strip() for item in texts):
        raise HTTPException(status_code=400, detail="all inputs must be non-empty strings")
    return texts


def create_app(settings: Settings | None = None, runtime: EmbedderRuntime | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_runtime = runtime or EmbedderRuntime(resolved_settings)
    app = FastAPI(title="Embedding Provider", version="0.2.0")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "service": resolved_settings.service_name,
            "model": resolved_settings.model_id,
            "model_alias": resolved_settings.model_alias,
            "task": resolved_settings.embedding_task,
            "dimensions": resolved_settings.default_dimensions,
            "device": resolved_runtime.device_name,
        }

    @app.get("/v1/models", response_model=ModelList)
    def list_models() -> ModelList:
        return ModelList(data=[ModelInfo(id=resolved_settings.model_alias or resolved_settings.model_id)])

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        if request.encoding_format not in (None, "float"):
            raise HTTPException(status_code=400, detail="Only float encoding_format is supported")
        allowed_models = {resolved_settings.model_id}
        if resolved_settings.model_alias:
            allowed_models.add(resolved_settings.model_alias)
        if request.model and request.model not in allowed_models:
            raise HTTPException(
                status_code=400,
                detail=f"Loaded model is {resolved_settings.model_id}, got {request.model}",
            )
        texts = _validate_inputs(request.input)
        try:
            embeddings = resolved_runtime.encode(
                texts,
                dimensions=request.dimensions,
                task=request.task,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return EmbeddingResponse(
            data=[EmbeddingItem(index=idx, embedding=embedding) for idx, embedding in enumerate(embeddings)],
            model=resolved_settings.model_alias or resolved_settings.model_id,
            usage=EmbeddingUsage(
                prompt_tokens=_estimate_tokens(texts),
                total_tokens=_estimate_tokens(texts),
            ),
        )

    return app


app = create_app()
