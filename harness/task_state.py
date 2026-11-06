# -*- coding: utf-8 -*-
"""结构化任务状态（checkpoint）：给调度器 / Harness 看的"任务真相"，不是聊天记录。

存：原始目标、验收标准、待办清单、已完成动作、产物位置、未确认问题、环境指纹。
双写：SQLite（task_state + task_state_history 版本历史）+ data/tasks/<task_id>.json（人可读镜像）。

恢复时**不允许直接拿快照继续跑**：
    verify() 会去查真实现场（数据库行数、产物记录是否还在、内容哈希是否一致），
    一致 → 沿用；不一致 → 标记 unknown / needs_recheck，写进 open_questions，
    并把结论注入上下文，明确禁止模型沿用未经核实的历史进度。
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

from . import config

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_NEEDS_RECHECK = "needs_recheck"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Todo:
    id: int
    text: str
    status: str = "pending"      # pending / in_progress / done / blocked / unknown
    evidence: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class TaskState:
    task_id: str
    session_id: str
    goal: str
    acceptance: List[str] = field(default_factory=list)
    todos: List[Todo] = field(default_factory=list)
    completed_actions: List[Dict] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    status: str = STATUS_RUNNING
    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    env: Dict = field(default_factory=dict)
    verify_report: Dict = field(default_factory=dict)

    # ---------------------------------------------------------------- 序列化
    def to_dict(self) -> Dict:
        data = asdict(self)
        data["todos"] = [t.to_dict() if isinstance(t, Todo) else t for t in self.todos]
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "TaskState":
        data = dict(data)
        todos = [Todo(**t) if isinstance(t, dict) else t for t in data.get("todos", [])]
        data["todos"] = todos
        allowed = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in data.items() if k in allowed}
        return cls(**data)

    # ---------------------------------------------------------------- 变更
    def add_todo(self, text: str) -> Todo:
        todo = Todo(id=len(self.todos) + 1, text=text)
        self.todos.append(todo)
        return todo

    def set_todo(self, todo_id: int, status: str, note: str = "", evidence: List[str] = None):
        for todo in self.todos:
            if todo.id == todo_id:
                todo.status = status
                if note:
                    todo.note = note
                if evidence:
                    todo.evidence = list(dict.fromkeys(todo.evidence + evidence))
                return todo
        return None

    def record_action(self, tool: str, args: Dict, ok: bool, summary: str,
                      artifact: str = None, row_id: int = None):
        self.completed_actions.append({
            "seq": len(self.completed_actions) + 1,
            "tool": tool,
            "args": args,
            "ok": bool(ok),
            "summary": summary[:300],
            "artifact": artifact,
            "row_id": row_id,
            "at": _now(),
        })
        if artifact:
            self.add_artifact(artifact)
        if self.todos:
            # 有当前进行中的待办就顺手推进，没有则不动（由模型/调用方显式设置）
            for todo in self.todos:
                if todo.status == "in_progress" and ok:
                    todo.evidence = list(dict.fromkeys(todo.evidence + ([artifact] if artifact else [])))
                    break

    def add_artifact(self, ref: str):
        if ref and ref not in self.artifacts:
            self.artifacts.append(ref)

    def add_question(self, text: str):
        if text and text not in self.open_questions:
            self.open_questions.append(text)

    def bump(self):
        self.version += 1
        self.updated_at = _now()

    # ---------------------------------------------------------------- 展示
    def render(self, max_todos: int = 6, max_actions: int = 5) -> str:
        lines = [f"任务ID：{self.task_id}｜状态：{self.status}｜版本：v{self.version}"]
        lines.append(f"原始目标：{self.goal}")
        if self.acceptance:
            lines.append("验收标准：")
            lines += [f"  - {a}" for a in self.acceptance[:6]]
        if self.todos:
            lines.append("待办清单：")
            for t in self.todos[:max_todos]:
                mark = {"done": "✔", "in_progress": "▶", "blocked": "✖",
                        "unknown": "?", "pending": "·"}.get(t.status, "·")
                ev = f"（证据 {len(t.evidence)} 项）" if t.evidence else ""
                note = f" {t.note}" if t.note else ""
                lines.append(f"  {mark} #{t.id} {t.text} [{t.status}]{ev}{note}")
            if len(self.todos) > max_todos:
                lines.append(f"  …另有 {len(self.todos) - max_todos} 条")
        if self.completed_actions:
            lines.append(f"已完成动作（共 {len(self.completed_actions)} 次调用，最近 {min(max_actions, len(self.completed_actions))} 次）：")
            for a in self.completed_actions[-max_actions:]:
                flag = "成功" if a["ok"] else "失败"
                lines.append(f"  - [{flag}] {a['tool']} {json.dumps(a['args'], ensure_ascii=False)[:80]}")
        if self.artifacts:
            lines.append("产物位置：" + "；".join(self.artifacts[-4:]))
        if self.open_questions:
            lines.append("未确认问题：")
            lines += [f"  ! {q}" for q in self.open_questions[:5]]
        lines += self._render_verify()
        return "\n".join(lines)

    def _render_verify(self) -> List[str]:
        rep = self.verify_report or {}
        if not rep:
            return []
        out = ["[恢复校验] " + ("通过，快照与现场一致" if rep.get("ok")
                               else "存在不一致，相关进度已标记 unknown，禁止沿用未经核实的结果")]
        for item in rep.get("mismatches", [])[:3]:
            out.append(f"  ✗ {item}")
        for item in rep.get("unknown", [])[:3]:
            out.append(f"  ? {item}")
        if rep.get("env_note"):
            out.append(f"  · {rep['env_note']}")
        return out


class TaskStateManager:
    def __init__(self, store, tracer=None):
        self.store = store
        self.tracer = tracer
        config.ensure_dirs()

    # ---------------------------------------------------------------- 载入 / 新建
    def load_or_create(self, task_id: str, session_id: str, goal: str):
        row = self.store.state_load(task_id)
        if not row:
            state = TaskState(task_id=task_id, session_id=session_id, goal=goal,
                              created_at=_now(), updated_at=_now())
            state.acceptance = self._guess_acceptance(goal)
            state.env = self._safe_fingerprint()
            if self.tracer:
                self.tracer.event("state.create", task_id=task_id, goal=goal,
                                  acceptance=state.acceptance)
            return state, False

        state = TaskState.from_dict(json.loads(row["payload"]))
        state.session_id = session_id
        if self.tracer:
            self.tracer.event("state.load", task_id=task_id, version=state.version,
                              status=state.status, todos=len(state.todos),
                              actions=len(state.completed_actions),
                              artifacts=state.artifacts)
        return state, True

    @staticmethod
    def _guess_acceptance(goal: str) -> List[str]:
        """粗粒度拆验收标准（不调模型，保持 Harness 层可解释）。"""
        items = []
        if any(k in goal for k in ("新增", "添加", "录入", "插入")):
            items.append("目标记录已存在于数据库，且字段与用户要求一致")
        if any(k in goal for k in ("修改", "更新", "改成", "改为", "调整")):
            items.append("目标记录的目标字段已被更新为新值，其他字段未被误改")
        if any(k in goal for k in ("删除", "删掉", "移除")):
            items.append("目标记录已不存在，且删除的是用户指定的那一条")
        if any(k in goal for k in ("查", "多少", "几个", "看看", "统计")):
            items.append("回答中的记录与数据库当前状态一致（数量、id、分数）")
        if not items:
            items.append("用户请求被完整满足，且未产生副作用")
        items.append("所有写入操作都有可核对的记录 id 证据")
        return items

    # ---------------------------------------------------------------- 保存
    def save(self, state: TaskState):
        state.bump()
        payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
        self.store.state_save(state.task_id, state.session_id, state.goal,
                              state.status, state.version, payload)
        self._write_mirror(state, payload)
        if self.tracer:
            self.tracer.event("state.save", task_id=state.task_id, version=state.version,
                              status=state.status, todos=len(state.todos),
                              actions=len(state.completed_actions),
                              artifacts=state.artifacts[-3:])

    def _write_mirror(self, state: TaskState, payload: str):
        path = os.path.join(config.TASK_DIR, f"{state.task_id}.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        except Exception as exc:
            if self.tracer:
                self.tracer.event("state.mirror_error", error=str(exc), path=path)

    # ---------------------------------------------------------------- 恢复校验
    @staticmethod
    def _check_ref(ref: str, fetch_row, row_fingerprint, parse_artifact, table):
        """核对一条产物引用，返回 (ok|mismatch|unknown, 说明)。

        引用格式：
            student_scores#id=1@ab12cd34   —— 期望该行存在且内容哈希一致
            !student_scores#id=1           —— 期望该行已不存在（删除类产物）
        """
        expect_absent = ref.startswith("!")
        clean = ref[1:] if expect_absent else ref
        tbl, row_id, digest = parse_artifact(clean)
        if row_id is None:
            return "unknown", f"产物引用格式无法解析：{ref}"
        if tbl != table:
            return "unknown", f"产物属于其他表：{ref}"
        try:
            row = fetch_row(row_id)
        except Exception as exc:
            return "unknown", f"读取 {ref} 失败：{exc}"

        if expect_absent:
            if row is not None:
                return "mismatch", f"产物 {tbl}#id={row_id} 应已删除，但现场仍存在（快照与事实不符）"
            return "ok", f"产物 {tbl}#id={row_id} 已按快照删除，核对通过"

        if row is None:
            return "mismatch", f"产物 {tbl}#id={row_id} 已不存在（快照里有，现场没有）"
        if row_fingerprint(row) != digest:
            return "mismatch", (f"产物 {tbl}#id={row_id} 内容已被外部改动"
                                f"（快照哈希 {digest}，现场哈希 {row_fingerprint(row)}）")
        return "ok", f"产物 {tbl}#id={row_id} 与快照一致"

    def _safe_fingerprint(self) -> Dict:
        try:
            from tools import db_fingerprint
            return db_fingerprint()
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def verify(self, state: TaskState) -> Dict:
        """拿快照去核对真实现场。只读，不做任何修复动作。"""
        report = {"checked_at": _now(), "matches": [], "mismatches": [],
                  "unknown": [], "ok": True, "env_note": ""}

        # 1) 环境指纹
        current = self._safe_fingerprint()
        if "error" in current:
            report["ok"] = False
            report["unknown"].append(f"数据库不可达，无法核对任何进度：{current['error']}")
            report["env_note"] = "现场不可读，全部进度视为未知"
        elif state.env and "row_count" in state.env:
            if current.get("row_count") != state.env.get("row_count") or \
                    current.get("max_id") != state.env.get("max_id"):
                report["ok"] = False
                report["env_note"] = (f"表行数 {state.env.get('row_count')}→{current.get('row_count')}，"
                                      f"最大 id {state.env.get('max_id')}→{current.get('max_id')}，"
                                      f"环境在快照之后发生过变化")
                report["mismatches"].append("环境指纹变化：可能有人新增/删除了记录")
            else:
                report["matches"].append("环境指纹一致（行数/最大 id 未变）")
            state.env = current
        else:
            state.env = current

        # 2) 产物记录逐一核对
        from tools import fetch_row, row_fingerprint, parse_artifact, TABLE
        alive_ok = 0
        for ref in list(state.artifacts):
            kind, msg = self._check_ref(ref, fetch_row, row_fingerprint, parse_artifact, TABLE)
            if kind == "ok":
                alive_ok += 1
                report["matches"].append(msg)
            elif kind == "mismatch":
                report["mismatches"].append(msg)
            elif kind == "unknown":
                report["unknown"].append(msg)
        if alive_ok:
            report["env_note"] = (report["env_note"] + "；" if report["env_note"] else "") + \
                                 f"{alive_ok} 个产物核对通过"

        # 3) 已完成的待办如果证据不符 → 降级为 unknown
        stale_todos = []
        for todo in state.todos:
            if todo.status == "done" and todo.evidence:
                for ref in todo.evidence:
                    kind, _ = self._check_ref(ref, fetch_row, row_fingerprint,
                                              parse_artifact, TABLE)
                    if kind in {"mismatch", "unknown"}:
                        todo.status = "unknown"
                        todo.note = "证据已失效，需要重新核实"
                        stale_todos.append(todo.id)
                        break
        if stale_todos:
            report["ok"] = False
            report["mismatches"].append(f"待办 {stale_todos} 的证据已失效，状态降级为 unknown")

        if report["mismatches"] or report["unknown"]:
            report["ok"] = False
            state.status = STATUS_NEEDS_RECHECK
            for item in report["mismatches"]:
                state.add_question("恢复校验不一致：" + item)
            for item in report["unknown"]:
                state.add_question("恢复校验不确定：" + item)
        else:
            report["ok"] = True
            if state.status == STATUS_NEEDS_RECHECK:
                state.status = STATUS_RUNNING

        state.verify_report = report
        if self.tracer:
            self.tracer.event("state.verify", task_id=state.task_id, ok=report["ok"],
                              matches=len(report["matches"]),
                              mismatches=report["mismatches"],
                              unknown=report["unknown"], env=current)
        return report

    # ---------------------------------------------------------------- 查询
    def show(self, task_id: str) -> Optional[str]:
        row = self.store.state_load(task_id)
        if not row:
            return None
        return row["payload"]
