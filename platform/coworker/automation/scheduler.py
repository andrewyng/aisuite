"""The scheduler loop — runs in the always-on server.

Policy (agreed): **run-once-catch-up** for runs missed while down (due tasks fire once on
startup, then resume), and **skip-on-overlap** (don't stack a run if the previous is still
going). The actual execution is injected as `runner(task, trigger) -> TaskRun` so this stays
independent of the engine/manager.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from .models import ScheduledTask, TaskRun
from .store import TaskStore

logger = logging.getLogger("coworker.automation")

Runner = Callable[[ScheduledTask, str], Awaitable[TaskRun]]


class Scheduler:
    def __init__(
        self, store: TaskStore, runner: Runner, *, tick_seconds: float = 30.0
    ) -> None:
        self.store = store
        self.runner = runner
        self.tick_seconds = tick_seconds
        self._task: Optional[asyncio.Task] = None
        self._running_ids: set[str] = set()  # overlap guard

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        # First pass = run-once-catch-up for anything missed while the server was down.
        try:
            await self._tick(trigger="catchup")
        except Exception:
            logger.exception("scheduler catch-up failed")
        while True:
            await asyncio.sleep(self.tick_seconds)
            try:
                await self._tick(trigger="schedule")
            except Exception:
                logger.exception("scheduler tick failed")

    async def _tick(self, *, trigger: str) -> None:
        for task in self.store.due():
            await self.run_task(task, trigger=trigger)

    async def run_task(self, task: ScheduledTask, *, trigger: str) -> Optional[TaskRun]:
        if task.id in self._running_ids:  # skip-on-overlap
            logger.info("skipping %s — previous run still going", task.id)
            return None
        self._running_ids.add(task.id)
        try:
            run = await self.runner(task, trigger)
        except Exception as exc:
            logger.exception("task %s run failed", task.id)
            run = TaskRun(
                task_id=task.id, status="error", error=str(exc), trigger=trigger
            )
            self.store.add_run(run)
        finally:
            self._running_ids.discard(task.id)
        # advance the task (run_count/last_run) → save recomputes next_run.
        fresh = self.store.get(task.id)
        if fresh is not None:
            fresh.run_count += 1
            fresh.last_run = run.started_at if run else None
            fresh.last_status = run.status if run else "error"
            self.store.save(fresh)
        return run
