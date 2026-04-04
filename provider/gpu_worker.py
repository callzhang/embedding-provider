from __future__ import annotations

import json
import logging
import sys
from typing import Any

import numpy as np
import torch
from transformers import AutoModel

from provider.config import Settings

log = logging.getLogger("embedding_provider_worker")
logging.basicConfig(level="INFO")


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(name.lower(), torch.bfloat16)


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


def _load_model(settings: Settings) -> Any:
    kwargs: dict[str, Any] = {
        "trust_remote_code": settings.trust_remote_code,
        "torch_dtype": _resolve_dtype(settings.dtype),
    }
    if settings.attn_implementation:
        kwargs["attn_implementation"] = settings.attn_implementation
    try:
        model = AutoModel.from_pretrained(settings.model_id, **kwargs)
    except Exception:
        log.warning("model load with custom attention failed; retrying without attn_implementation", exc_info=True)
        kwargs.pop("attn_implementation", None)
        model = AutoModel.from_pretrained(settings.model_id, **kwargs)
    if hasattr(model, "to"):
        model = model.to(torch.device("cuda"))
    if hasattr(model, "eval"):
        model.eval()
    return model


def main() -> int:
    settings = Settings.from_env()
    try:
        model = _load_model(settings)
        encode_signature = __import__("inspect").signature(model.encode)
    except Exception as exc:
        _emit({"status": "error", "error": repr(exc)})
        return 1

    _emit({"status": "ready", "device": "cuda", "model": settings.model_id})

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        request = json.loads(line)
        op = str(request.get("op") or "")
        if op == "shutdown":
            _emit({"status": "ok"})
            return 0
        if op != "encode":
            _emit({"status": "error", "error": f"unsupported op: {op}"})
            continue
        texts = [str(item) for item in request.get("texts") or []]
        dimensions = request.get("dimensions")
        task = request.get("task") or settings.embedding_task
        kwargs: dict[str, Any] = {}
        input_name = "texts" if "texts" in encode_signature.parameters else "sentences"
        if task and "task" in encode_signature.parameters:
            kwargs["task"] = str(task)
        if dimensions and "truncate_dim" in encode_signature.parameters:
            kwargs["truncate_dim"] = int(dimensions)
        if settings.max_length and "max_length" in encode_signature.parameters:
            kwargs["max_length"] = settings.max_length
        sample_bytes_per_text = None
        try:
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
            with torch.no_grad():
                outputs = model.encode(**{input_name: texts}, **kwargs)
            delta = torch.cuda.max_memory_allocated() - mem_before
            if delta > 0 and texts:
                sample_bytes_per_text = float(delta / len(texts))
            array = _outputs_to_numpy(outputs)
            if array.ndim == 1:
                array = np.expand_dims(array, axis=0)
            if dimensions and "truncate_dim" not in encode_signature.parameters:
                requested = int(dimensions)
                if requested > array.shape[1]:
                    raise ValueError(f"Requested dimensions={requested} but model only produced {array.shape[1]}")
                array = array[:, :requested]
            if settings.normalize_embeddings:
                norms = np.linalg.norm(array, axis=1, keepdims=True)
                norms = np.where(norms == 0.0, 1.0, norms)
                array = array / norms
            _emit(
                {
                    "status": "ok",
                    "embeddings": array.tolist(),
                    "sample_bytes_per_text": sample_bytes_per_text,
                }
            )
        except Exception as exc:
            _emit({"status": "error", "error": repr(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
