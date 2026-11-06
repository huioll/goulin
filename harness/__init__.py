# -*- coding: utf-8 -*-
"""Harness：三层记忆 + ReAct 循环的粘合层。

记忆体系分区（严格区分用途）：
    short_term  —— 当前会话的轨迹，预算内降级（去重→剔低相关→摘要压缩）
    task_state  —— 结构化 checkpoint，给调度器/Harness 用，恢复时必须核对现场
    long_term   —— 跨会话向量记忆，只作参考证据，带 TTL/来源/矛盾抑制

对外暴露：Store / Tracer / Embedder / ShortTermMemory / TaskStateManager / LongTermMemory / ContextAssembler
"""

from . import config
from .context import ContextAssembler, ContextPackage
from .embedding import Embedder
from .long_term import LongTermMemory, MemoryHit
from .short_term import ShortTermMemory, STMBlock
from .store import Store
from .task_state import TaskState, TaskStateManager, Todo
from .trace import Tracer

__all__ = [
    "config", "Store", "Tracer", "Embedder", "ShortTermMemory", "STMBlock",
    "TaskStateManager", "TaskState", "Todo", "LongTermMemory", "MemoryHit",
    "ContextAssembler", "ContextPackage",
]
