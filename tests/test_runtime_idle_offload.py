from __future__ import annotations

import os
import sys
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

if "torch" not in sys.modules:
    fake_torch = types.ModuleType("torch")

    class _FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTensor:
        pass

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def mem_get_info():
            return (8 * 1024**3, 24 * 1024**3)

        @staticmethod
        def reset_peak_memory_stats() -> None:
            return None

        @staticmethod
        def memory_allocated() -> int:
            return 0

        @staticmethod
        def max_memory_allocated() -> int:
            return 0

    fake_torch.cuda = _FakeCuda()
    fake_torch.no_grad = _FakeNoGrad
    fake_torch.device = lambda name: name
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float32 = "float32"
    fake_torch.Tensor = _FakeTensor
    fake_torch.OutOfMemoryError = RuntimeError
    sys.modules["torch"] = fake_torch

if "transformers" not in sys.modules:
    fake_transformers = types.ModuleType("transformers")

    class _BootstrapModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def encode(self, texts: list[str], **kwargs) -> np.ndarray:
            width = kwargs.get("truncate_dim") or 4
            return np.ones((len(texts), width), dtype=np.float32)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _BootstrapModel()

    fake_transformers.AutoModel = _FakeAutoModel
    sys.modules["transformers"] = fake_transformers

from provider.app import EmbedderRuntime
from provider.config import Settings


class FakeModel:
    def __init__(self, load_device: str) -> None:
        self.load_device = load_device
        self.to_device = load_device

    def to(self, device) -> "FakeModel":
        self.to_device = str(device)
        return self

    def eval(self) -> "FakeModel":
        return self

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        width = kwargs.get("truncate_dim") or 4
        return np.ones((len(texts), width), dtype=np.float32)


class EmbedderRuntimeIdleOffloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_stack = ExitStack()
        env = {
            "SERVICE_NAME": "embedding-provider",
            "MODEL_ID": "jinaai/jina-embeddings-v5-text-nano",
            "MODEL_ALIAS": "jinaai/jina-embeddings-v5-text-nano",
            "EMBEDDING_TASK": "text-matching",
            "DEFAULT_DIMENSIONS": "4",
            "MAX_LENGTH": "256",
            "MAX_BATCH_SIZE": "8",
            "BATCH_WINDOW_MS": "200",
            "IDLE_OFFLOAD_SECONDS": "1",
            "IDLE_OFFLOAD_POLL_SECONDS": "1",
            "NORMALIZE_EMBEDDINGS": "true",
            "DTYPE": "float32",
            "TRUST_REMOTE_CODE": "true",
        }
        for key, value in env.items():
            self._env_stack.enter_context(patch.dict(os.environ, {key: value}))

    def tearDown(self) -> None:
        self._env_stack.close()

    def test_idle_runtime_offloads_and_reloads(self) -> None:
        built_devices: list[str] = []

        def fake_from_pretrained(*args, **kwargs):
            built_devices.append("loaded")
            return FakeModel(load_device="cpu")

        with (
            patch("provider.app.AutoModel.from_pretrained", side_effect=fake_from_pretrained),
            patch("provider.app.torch.cuda.is_available", return_value=True),
            patch("provider.app.torch.cuda.empty_cache"),
            patch("provider.app.torch.cuda.mem_get_info", return_value=(8 * 1024**3, 24 * 1024**3)),
        ):
            runtime = EmbedderRuntime(Settings.from_env())
            self.assertEqual(runtime.runtime_status()["loaded_device"], "cuda")

            runtime._last_encode_finished_at -= 2
            offloaded = runtime.maybe_offload_idle()
            self.assertTrue(offloaded)
            self.assertEqual(runtime.runtime_status()["loaded_device"], "cpu")
            self.assertEqual(runtime.runtime_status()["engine_state"], "offloaded")

            embeddings = runtime.encode(["hello world"])
            self.assertEqual(len(embeddings), 1)
            self.assertEqual(runtime.runtime_status()["loaded_device"], "cuda")
            self.assertEqual(runtime.runtime_status()["engine_state"], "hot")

        self.assertEqual(len(built_devices), 3)


if __name__ == "__main__":
    unittest.main()
