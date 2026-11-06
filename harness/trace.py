# -*- coding: utf-8 -*-
"""可观测追踪：每次记忆读写 / 召回 / 裁剪 / LLM 调用 / 工具执行都落 trace。

双写：SQLite（便于按 task 查询）+ JSONL（便于 grep / 外部工具分析）。
事件命名用 "域.动作"，方便排查：
    stm.read / stm.write / stm.degrade / stm.dedupe / stm.compress
    ltm.recall / ltm.write / ltm.supersede / ltm.skip
    state.load / state.save / state.verify
    ctx.assemble / llm.call / tool.exec / react.step / react.finish
"""

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime

from . import config


class Tracer:
    def __init__(self, store, session_id: str = None, task_id: str = None, echo: bool = False):
        self.store = store
        self.session_id = session_id
        self.task_id = task_id
        self.step = 0
        self.echo = echo
        config.ensure_dirs()
        self.jsonl_path = os.path.join(
            config.TRACE_JSONL_DIR, f"trace-{datetime.now():%Y%m%d}.jsonl"
        )
        self.buffer = []   # 内存里保留本次运行的尾部，方便收尾时打印摘要

    def bind(self, session_id: str = None, task_id: str = None, step: int = None):
        if session_id is not None:
            self.session_id = session_id
        if task_id is not None:
            self.task_id = task_id
        if step is not None:
            self.step = step

    def event(self, name: str, duration_ms: float = None, **detail):
        payload = {k: v for k, v in detail.items()}
        try:
            self.store.trace_add(self.session_id, self.task_id, self.step, name,
                                 payload, duration_ms)
        except Exception as exc:      # trace 不能影响主流程
            payload = dict(payload, trace_error=str(exc))
        record = {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session": self.session_id,
            "task": self.task_id,
            "step": self.step,
            "event": name,
            "detail": payload,
            "ms": round(duration_ms, 2) if duration_ms is not None else None,
        }
        self.buffer.append(record)
        if len(self.buffer) > 500:
            self.buffer = self.buffer[-500:]
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass
        if self.echo:
            brief = json.dumps(payload, ensure_ascii=False, default=str)
            print(f"    [trace] {name} {brief[:220]}")

    @contextmanager
    def span(self, name: str, **detail):
        """记录耗时；块内可以通过 yield 出来的 dict 追加结果信息。"""
        extra = {}
        started = time.perf_counter()
        try:
            yield extra
        except Exception as exc:
            extra.setdefault("error", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            cost = (time.perf_counter() - started) * 1000
            merged = dict(detail, **extra)
            self.event(name, duration_ms=cost, **merged)

    def summary(self, limit: int = 12):
        """返回本次运行最近的事件摘要（控制台收尾用）。"""
        return self.buffer[-limit:]
