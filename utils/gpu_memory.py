from __future__ import annotations

import gc
import logging

try:
    import torch
except Exception:  # pragma: no cover - optional dependency on local CPU machine
    torch = None


def log_gpu_memory(logger: logging.Logger | None = None, prefix: str = "GPU") -> None:
    if torch is None or not torch.cuda.is_available():
        if logger is not None:
            logger.debug("%s memory: CUDA unavailable", prefix)
        return

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    message = f"{prefix} memory allocated={allocated:.2f}MB reserved={reserved:.2f}MB"
    if logger is not None:
        logger.info(message)


def clear_cuda_cache(logger: logging.Logger | None = None, reason: str = "manual") -> None:
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if logger is not None:
        logger.info("Cleared Python/CUDA cache (%s).", reason)

