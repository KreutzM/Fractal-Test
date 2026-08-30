from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Callable, Generic, TypeVar


ResultT = TypeVar("ResultT")


class RenderController(Generic[ResultT]):
    """Own the single-worker render lifecycle and coalesce invalidations."""

    def __init__(self, executor: Executor | None = None) -> None:
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="fractal-render",
        )
        self._owns_executor = executor is None
        self._generation = 0
        self._active_generation: int | None = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def busy(self) -> bool:
        return self._active_generation is not None

    def invalidate(self) -> int:
        self._generation += 1
        return self._generation

    def submit(self, job: Callable[[], ResultT]) -> tuple[int, Future[ResultT]] | None:
        if self.busy:
            return None
        generation = self._generation
        self._active_generation = generation
        try:
            future = self._executor.submit(job)
        except Exception:
            self._active_generation = None
            raise
        return generation, future

    def complete(self, generation: int) -> bool:
        if self._active_generation == generation:
            self._active_generation = None
        return generation != self._generation

    def shutdown(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
