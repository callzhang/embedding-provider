from __future__ import annotations

import asyncio
import functools
import gc
import inspect
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
import threading
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
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


class _GpuEmbedderWorker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._root_dir = Path(__file__).resolve().parents[1]
        self._process: subprocess.Popen[str] | None = None
        self._io_lock = threading.RLock()
        self._device_name = "none"

    @property
    def pid(self) -> int | None:
        if self._process is None:
            return None
        return self._process.pid

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_started(self) -> str:
        with self._io_lock:
            if self.is_running():
                return self._device_name
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self._process = subprocess.Popen(
                [sys.executable, "-m", "provider.gpu_worker"],
                cwd=str(self._root_dir),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            message = self._read_message()
            if str(message.get("status")) != "ready":
                self.terminate()
                raise RuntimeError(str(message.get("error") or "failed to start embedding GPU worker"))
            self._device_name = str(message.get("device") or "cuda")
            return self._device_name

    def encode(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        task: str | None = None,
    ) -> tuple[list[list[float]], float | None]:
        response = self._request(
            {
                "op": "encode",
                "texts": texts,
                "dimensions": dimensions,
                "task": task,
            }
        )
        embeddings = [[float(value) for value in row] for row in response.get("embeddings") or []]
        sample = response.get("sample_bytes_per_text")
        return embeddings, float(sample) if sample is not None else None

    def terminate(self) -> None:
        with self._io_lock:
            process = self._process
            self._process = None
            self._device_name = "none"
            if process is None:
                return
            if process.poll() is None:
                try:
                    self._write_message({"op": "shutdown"}, process=process)
                    process.wait(timeout=2)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except Exception:
                        process.kill()
                        process.wait(timeout=5)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._io_lock:
            self.ensure_started()
            self._write_message(payload)
            response = self._read_message()
        if str(response.get("status")) != "ok":
            raise RuntimeError(str(response.get("error") or "embedding GPU worker request failed"))
        return response

    def _write_message(self, payload: dict[str, Any], *, process: subprocess.Popen[str] | None = None) -> None:
        target = process or self._process
        if target is None or target.stdin is None:
            raise RuntimeError("embedding GPU worker stdin is unavailable")
        target.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        target.stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("embedding GPU worker stdout is unavailable")
        line = self._process.stdout.readline()
        if not line:
            return {"status": "error", "error": "embedding GPU worker exited before responding"}
        return json.loads(line)


class EmbedderRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._preferred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = self._preferred_device
        self.device_name = "none" if str(self._preferred_device) == "cuda" else str(self._device)
        self._model_lock = threading.Condition()
        self._use_gpu_worker = str(self._preferred_device) == "cuda"
        self._gpu_worker: _GpuEmbedderWorker | None = _GpuEmbedderWorker(settings) if self._use_gpu_worker else None
        self._model = None if self._use_gpu_worker else self._load_model()
        self._encode_signature = None if self._model is None else inspect.signature(self._model.encode)
        self._bytes_per_text_ema: float | None = None
        self._ema_alpha = 0.25
        self._engine_state = "offloaded" if self._use_gpu_worker else "hot"
        self._reload_in_progress = False
        self._inflight_encodes = 0
        self._last_encode_finished_at = time.monotonic()
        self._last_offloaded_at: float | None = None
        self._idle_offload_enabled = self._use_gpu_worker and settings.idle_offload_seconds > 0

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

    def _load_model(self, device_override: torch.device | None = None) -> Any:
        target_device = device_override or self._preferred_device
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
            model = model.to(target_device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    def runtime_status(self) -> dict[str, Any]:
        with self._model_lock:
            idle_for = max(0.0, time.monotonic() - self._last_encode_finished_at)
            offloaded_for = None
            if self._last_offloaded_at is not None:
                offloaded_for = max(0.0, time.monotonic() - self._last_offloaded_at)
            return {
                "loaded_device": self.device_name,
                "preferred_device": str(self._preferred_device),
                "engine_state": self._engine_state,
                "idle_offload_enabled": self._idle_offload_enabled,
                "idle_offload_seconds": self._settings.idle_offload_seconds if self._idle_offload_enabled else None,
                "idle_offload_poll_seconds": (
                    self._settings.idle_offload_poll_seconds if self._idle_offload_enabled else None
                ),
                "inflight_encodes": self._inflight_encodes,
                "idle_for_seconds": round(idle_for, 3),
                "offloaded_for_seconds": round(offloaded_for, 3) if offloaded_for is not None else None,
                "reload_in_progress": self._reload_in_progress,
                "worker_pid": self._gpu_worker.pid if self._gpu_worker is not None and self._gpu_worker.is_running() else None,
            }

    def close(self) -> None:
        if self._gpu_worker is not None:
            self._gpu_worker.terminate()

    def maybe_offload_idle(self) -> bool:
        if not self._idle_offload_enabled:
            return False
        with self._model_lock:
            if self._reload_in_progress or self._engine_state == "offloaded" or self._inflight_encodes > 0:
                return False
            idle_for = time.monotonic() - self._last_encode_finished_at
            if idle_for < self._settings.idle_offload_seconds:
                return False
            self._reload_in_progress = True
        try:
            if self._use_gpu_worker:
                if self._gpu_worker is not None:
                    self._gpu_worker.terminate()
                with self._model_lock:
                    self._device = torch.device("cpu")
                    self.device_name = "none"
                    self._engine_state = "offloaded"
                    self._last_offloaded_at = time.monotonic()
            else:
                next_model = self._load_model(device_override=torch.device("cpu"))
                self._swap_model(next_model, torch.device("cpu"), engine_state="offloaded")
            return True
        finally:
            with self._model_lock:
                self._reload_in_progress = False
                self._model_lock.notify_all()

    def _ensure_hot_model(self) -> None:
        if not self._idle_offload_enabled:
            return
        with self._model_lock:
            while self._reload_in_progress:
                self._model_lock.wait()
            if self._engine_state == "hot":
                return
            self._reload_in_progress = True
        try:
            if self._use_gpu_worker:
                if self._gpu_worker is None:
                    raise RuntimeError("embedding GPU worker is unavailable")
                device_name = self._gpu_worker.ensure_started()
                with self._model_lock:
                    self._device = torch.device(device_name)
                    self.device_name = device_name
                    self._engine_state = "hot"
                    self._last_offloaded_at = None
            else:
                next_model = self._load_model(device_override=self._preferred_device)
                self._swap_model(next_model, self._preferred_device, engine_state="hot")
        finally:
            with self._model_lock:
                self._reload_in_progress = False
                self._model_lock.notify_all()

    def _begin_encode(self) -> None:
        with self._model_lock:
            self._inflight_encodes += 1
            self._model_lock.notify_all()
        try:
            self._ensure_hot_model()
        except Exception:
            self._finish_encode()
            raise

    def _finish_encode(self) -> None:
        with self._model_lock:
            self._inflight_encodes = max(0, self._inflight_encodes - 1)
            self._last_encode_finished_at = time.monotonic()
            self._model_lock.notify_all()

    def _swap_model(self, next_model: Any, next_device: torch.device, *, engine_state: str) -> None:
        if self._use_gpu_worker:
            raise RuntimeError("_swap_model is unavailable in GPU worker mode")
        with self._model_lock:
            old_model = self._model
            self._model = next_model
            self._encode_signature = inspect.signature(self._model.encode)
            self._device = next_device
            self.device_name = str(next_device)
            self._engine_state = engine_state
            if engine_state == "offloaded":
                self._last_offloaded_at = time.monotonic()
            else:
                self._last_offloaded_at = None
        if old_model is not next_model:
            del old_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _encode_once(self, texts: list[str], *, dimensions: int | None = None, task: str | None = None) -> list[list[float]]:
        if self._use_gpu_worker:
            if self._gpu_worker is None:
                raise RuntimeError("embedding GPU worker is unavailable")
            device_name = self._gpu_worker.ensure_started()
            with self._model_lock:
                self._device = torch.device(device_name)
                self.device_name = device_name
                self._engine_state = "hot"
                self._last_offloaded_at = None
            embeddings, sample = self._gpu_worker.encode(texts, dimensions=dimensions, task=task)
            if sample is not None and sample > 0:
                if self._bytes_per_text_ema is None:
                    self._bytes_per_text_ema = sample
                else:
                    self._bytes_per_text_ema = self._ema_alpha * sample + (1 - self._ema_alpha) * self._bytes_per_text_ema
            return embeddings
        kwargs: dict[str, Any] = {}
        if self._encode_signature is None:
            raise RuntimeError("embedding model signature is unavailable")
        input_name = "texts" if "texts" in self._encode_signature.parameters else "sentences"
        requested_dimensions = dimensions or self._settings.default_dimensions
        effective_task = task or self._settings.embedding_task
        if effective_task and "task" in self._encode_signature.parameters:
            kwargs["task"] = effective_task
        if requested_dimensions and "truncate_dim" in self._encode_signature.parameters:
            kwargs["truncate_dim"] = requested_dimensions
        if self._settings.max_length and "max_length" in self._encode_signature.parameters:
            kwargs["max_length"] = self._settings.max_length

        track_mem = torch.cuda.is_available()
        if track_mem:
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.encode(**{input_name: texts}, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if track_mem:
            delta = torch.cuda.max_memory_allocated() - mem_before
            if delta > 0:
                sample = delta / len(texts)
                if self._bytes_per_text_ema is None:
                    self._bytes_per_text_ema = sample
                else:
                    self._bytes_per_text_ema = self._ema_alpha * sample + (1 - self._ema_alpha) * self._bytes_per_text_ema

        log.info(
            "model=%s texts=%s elapsed_ms=%.1f bytes_per_text=%.0f",
            self._settings.model_id, len(texts), elapsed_ms,
            self._bytes_per_text_ema or 0,
        )

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
        self._begin_encode()
        try:
            max_batch_size = self._settings.max_batch_size or len(texts)
            embeddings: list[list[float]] = []
            for chunk in _batched(texts, max_batch_size):
                embeddings.extend(self._encode_with_backoff(chunk, dimensions=dimensions, task=task))
            return embeddings
        finally:
            self._finish_encode()

    def estimate_max_texts(self) -> int:
        """Return a safe upper bound on texts for the next forward pass based on free VRAM."""
        hard_cap = self._settings.max_batch_size
        if not torch.cuda.is_available():
            return hard_cap or 256

        free, total = torch.cuda.mem_get_info()
        # Reserve 512 MB + 5% of total as headroom for CUDA context, cuBLAS workspace, etc.
        safety = int(512 * 1024 * 1024) + int(total * 0.05)
        usable = max(0, free - safety)

        if self._bytes_per_text_ema is None or self._bytes_per_text_ema <= 0:
            # No measurements yet — conservative default until EMA warms up.
            estimate = 32
        else:
            estimate = max(1, int(usable / self._bytes_per_text_ema))

        if hard_cap:
            estimate = min(estimate, hard_cap)
        return max(1, estimate)


@dataclass
class _PendingItem:
    texts: list[str]
    dimensions: int | None
    task: str | None
    future: asyncio.Future  # resolved with list[list[float]]


class ContinuousBatcher:
    """Collects requests within a time window and dispatches them as a single batch.

    The worker loop is sequential: it collects a window of requests, runs one
    GPU forward pass (per task/dimensions group), resolves all futures, then
    starts the next window. This keeps GPU jobs serialized while requests queue
    up naturally during inference.
    """

    def __init__(self, runtime: EmbedderRuntime, window_secs: float) -> None:
        self._runtime = runtime
        self._window = window_secs
        self._queue: asyncio.Queue[_PendingItem] = asyncio.Queue()

    def start(self) -> asyncio.Task:
        return asyncio.create_task(self._worker())

    async def encode(
        self,
        texts: list[str],
        *,
        dimensions: int | None = None,
        task: str | None = None,
    ) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[list[float]]] = loop.create_future()
        await self._queue.put(_PendingItem(texts=texts, dimensions=dimensions, task=task, future=future))
        return await future

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        pending: list[_PendingItem] = []
        while True:
            if not pending:
                # Block until at least one request arrives.
                pending.append(await self._queue.get())

                # Drain for the remaining window.
                deadline = loop.time() + self._window
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                        pending.append(item)
                    except (asyncio.TimeoutError, TimeoutError):
                        break

            # Dispatch; overflow (texts that didn't fit in VRAM) is returned
            # and processed immediately in the next iteration without a new window wait.
            pending = await self._dispatch(pending, loop)

    async def _dispatch(self, batch: list[_PendingItem], loop: asyncio.AbstractEventLoop) -> list[_PendingItem]:
        # Group by (task, dimensions) — in practice almost always one group.
        groups: dict[tuple, list[_PendingItem]] = defaultdict(list)
        for item in batch:
            groups[(item.task, item.dimensions)].append(item)

        overflow: list[_PendingItem] = []

        for (task, dimensions), items in groups.items():
            # Cap this group by available VRAM; overflow is deferred to next iteration.
            max_texts = self._runtime.estimate_max_texts()
            to_process: list[_PendingItem] = []
            text_count = 0
            for item in items:
                if to_process and text_count + len(item.texts) > max_texts:
                    overflow.append(item)
                else:
                    to_process.append(item)
                    text_count += len(item.texts)

            all_texts: list[str] = []
            offsets: list[int] = []
            for item in to_process:
                offsets.append(len(all_texts))
                all_texts.extend(item.texts)

            log.info(
                "batch dispatch: requests=%d texts=%d overflow=%d task=%s dim=%s",
                len(to_process), text_count, len(items) - len(to_process), task, dimensions,
            )
            fn = functools.partial(self._runtime.encode, all_texts, dimensions=dimensions, task=task)
            try:
                embeddings: list[list[float]] = await loop.run_in_executor(None, fn)
                for item, offset in zip(to_process, offsets):
                    item.future.set_result(embeddings[offset: offset + len(item.texts)])
            except Exception as exc:
                for item in to_process:
                    if not item.future.done():
                        item.future.set_exception(exc)

        return overflow


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


def _require_api_key(settings: Settings, authorization: str | None) -> None:
    if not settings.api_key:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def create_app(settings: Settings | None = None, runtime: EmbedderRuntime | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_runtime = runtime or EmbedderRuntime(resolved_settings)
    batcher = ContinuousBatcher(resolved_runtime, window_secs=resolved_settings.batch_window_ms / 1000)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = batcher.start()
        offload_task: asyncio.Task | None = None
        if resolved_runtime.runtime_status()["idle_offload_enabled"]:
            async def idle_offload_worker() -> None:
                while True:
                    await asyncio.sleep(resolved_settings.idle_offload_poll_seconds)
                    await asyncio.get_running_loop().run_in_executor(None, resolved_runtime.maybe_offload_idle)

            offload_task = asyncio.create_task(idle_offload_worker())
        yield
        task.cancel()
        if offload_task is not None:
            offload_task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if offload_task is not None:
            try:
                await offload_task
            except asyncio.CancelledError:
                pass
        resolved_runtime.close()

    app = FastAPI(title="Embedding Provider", version="0.2.0", lifespan=lifespan)

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
            "batch_window_ms": resolved_settings.batch_window_ms,
            "runtime": resolved_runtime.runtime_status(),
        }

    @app.get("/v1/models", response_model=ModelList)
    def list_models(authorization: str | None = Header(default=None)) -> ModelList:
        _require_api_key(resolved_settings, authorization)
        return ModelList(data=[ModelInfo(id=resolved_settings.model_alias or resolved_settings.model_id)])

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(
        request: EmbeddingRequest,
        authorization: str | None = Header(default=None),
    ) -> EmbeddingResponse:
        _require_api_key(resolved_settings, authorization)
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
            embeddings = await batcher.encode(
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
