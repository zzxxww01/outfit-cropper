from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable, Type


def run_with_retries(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 2,
    base_delay_seconds: float = 1.0,
    exceptions: Iterable[Type[BaseException]] = (Exception,),
    logger: logging.Logger | None = None,
    step_name: str = "step",
    **kwargs: Any,
) -> Any:
    allowed_exceptions = tuple(exceptions)
    total_attempts = 1 + max_retries
    attempt = 0
    while attempt < total_attempts:
        attempt += 1
        try:
            return fn(*args, **kwargs)
        except allowed_exceptions as exc:
            if attempt >= total_attempts:
                raise
            sleep_seconds = base_delay_seconds * (2 ** (attempt - 1))
            if logger is not None:
                import traceback
                logger.warning(
                    "%s failed on attempt %s/%s: %s. Retrying in %.1fs",
                    step_name,
                    attempt,
                    total_attempts,
                    exc,
                    sleep_seconds,
                )
                logger.debug("Full traceback:\n%s", traceback.format_exc())
            time.sleep(sleep_seconds)

