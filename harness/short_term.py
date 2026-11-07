# -*- coding: utf-8 -*-
"""短期记忆：只管"当前会话"。

存什么：用户输入、每一轮的 Thought / Action / Observation、最终回答、压缩摘要。
超出 token 预算时按固定顺序降级（每一步都写 trace，绝不直接截断丢关键信息）：

    1. 去重      —— 相似度 ≥ 阈值的重复记录合并（保留最新一条，记录合并次数）
    2. 剔除低相关 —— 用当前任务做相关度打分，剔除低分的**非保护**记录
    3. 摘要压缩   —— 把老旧片段压成"决策 + 证据"摘要行（保留工具名、成功与否、记录 id）
    4. 兜底收缩   —— 极端情况下先压摘要行长度，再动最后的保底窗口

"保护记录"（protected）指：用户指令、写操作及其 Observation、失败结果、最终结论。
这些是"关键决策和证据"，去重之外不会被剔除，只会被压缩成含证据的摘要行。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config
from .tokens import clip_chars, estimate_tokens

PROTECTED_KINDS = {"user", "answer"}
KIND_LABEL = {
    "user": "用户", "thought": "思考", "action": "动作", "observation": "观察",
    "answer": "结论", "summary": "摘要", "note": "备注",
}
KIND_WEIGHT = {"user": 1.0, "thought": 0.8, "action": 1.0, "observation": 1.0,
               "answer": 1.0, "summary": 0.9, "note": 0.5}


@dataclass
class Record:
    id: int
    kind: str
    content: str
    protected: bool
    turn: int
    step: int
    tokens: int
    ts: str
    meta: Dict = field(default_factory=dict)
    merged: int = 0          # 被合并进来的重复条数


@dataclass
class STMBlock:
    text: str
    used_tokens: int
    before_tokens: int
    stages: List[Dict]
    kept_evidence: List[str]
    dropped_ids: List[int]
    rendered: int

    def brief(self) -> str:
        parts = [f"{s['stage']}:{s.get('count', '-')}条/{s['tokens']}tok" for s in self.stages]
        return f"{self.before_tokens}→{self.used_tokens} tok | " + " → ".join(parts)


class ShortTermMemory:
    def __init__(self, store, session_id: str, task_id: str = None, tracer=None):
        self.store = store
        self.session_id = session_id
        self.task_id = task_id
        self.tracer = tracer
        self.turn = 0
        self.step = 0
        self.records: List[Record] = []
        self.load()

    # ------------------------------------------------------------------ 读写
    def load(self):
        rows = self.store.stm_list(self.session_id)
        self.records = [self._row_to_record(r) for r in rows]
        if self.records:
            self.turn = max(r.turn for r in self.records)
        if self.tracer:
            self.tracer.event("stm.load", session=self.session_id,
                              records=len(self.records),
                              tokens=sum(r.tokens for r in self.records))

    @staticmethod
    def _row_to_record(row) -> Record:
        import json
        try:
            meta = json.loads(row.get("meta") or "{}")
        except Exception:
            meta = {}
        return Record(id=row["id"], kind=row["kind"], content=row["content"],
                      protected=bool(row["protected"]), turn=row["turn"], step=row["step"],
                      tokens=row["tokens"], ts=row["ts"], meta=meta)

    def _auto_protect(self, kind: str, meta: Dict) -> bool:
        if kind in PROTECTED_KINDS:
            return True
        if kind == "observation":
            # 写操作的结果、以及任何失败结果，都是后续决策要引用的证据
            if meta.get("success") is False:
                return True
            if meta.get("tool") in {"add_student", "update_student", "delete_student"}:
                return True
        return False

    def add(self, kind: str, content: str, protected: Optional[bool] = None,
            meta: Dict = None, turn: int = None, step: int = None) -> Record:
        meta = meta or {}
        protected = self._auto_protect(kind, meta) if protected is None else protected
        turn = self.turn if turn is None else turn
        step = self.step if step is None else step
        content = str(content)
        tokens = estimate_tokens(content)
        rec_id = self.store.stm_add(self.session_id, self.task_id, turn, step, kind,
                                    content, protected, meta, tokens)
        rec = Record(id=rec_id, kind=kind, content=content, protected=protected,
                     turn=turn, step=step, tokens=tokens,
                     ts="", meta=meta)
        self.records.append(rec)
        if self.tracer:
            self.tracer.event("stm.write", kind=kind, protected=protected, step=step,
                              tokens=tokens, preview=clip_chars(content, 80),
                              meta=meta)
        return rec

    def begin_turn(self, turn: int = None):
        self.turn = (self.turn + 1) if turn is None else turn
        self.step = 0

    def set_step(self, step: int):
        self.step = step

    # ------------------------------------------------------------------ 渲染
    def _line(self, idx: int, rec: Record, latest: bool = False) -> str:
        label = KIND_LABEL.get(rec.kind, rec.kind)
        # 最新一条观测就是本轮的决策依据，按工具层已限定的上限（MAX_OBS_CHARS）完整给；
        # 历史痕迹压到 220/300，避免一条旧结果挤占整块预算。
        # 一次 6 行班级查询 ≈ 1000 字符，之前限 900 会把尾部行切掉，
        # 模型只好逐个 id 补查，8 步内跑不完（评测 T04 踩过）。
        limit = config.MAX_OBS_CHARS if latest else (300 if rec.protected else 220)
        raw = rec.content.replace("\n", " ⏎ ")
        if len(raw) <= limit:
            body = raw
        else:
            # 截断必须显式标注：否则模型会把半截 JSON 当成完整结果——
            # 例如 {"count": 6} 却只看到 3 行，就误判"这个班只有 3 个人"（评测 T04/T08 踩过）。
            body = raw[:limit] + f" …（已截断，原长 {len(raw)} 字符；要完整数据请缩小查询范围或分页查）"
        prefix = ""
        if rec.kind == "observation" and rec.meta.get("tool"):
            # 观察一定要带上"哪个工具 + 成功/失败"，否则压缩后只剩一段 JSON，模型认不出证据归属
            ok = "成功" if rec.meta.get("success") else "失败"
            prefix = f"{rec.meta['tool']} {ok}｜"
        elif rec.kind == "action" and rec.meta.get("tool"):
            prefix = ""
        suffix = f"（合并{rec.merged}条重复）" if rec.merged else ""
        return f"{idx}. [{label}]{suffix} {prefix}{body}"

    def render(self, recs: List[Record]) -> str:
        if not recs:
            return "（本轮暂无执行轨迹）"
        total = len(recs)
        return "\n".join(self._line(i, r, latest=(i == total))   # i 从 1 开始，最后一条是 total
                         for i, r in enumerate(recs, 1))

    @staticmethod
    def _terms(text: str) -> set:
        import re
        text = (text or "").lower()
        chars = re.findall(r"[\u4e00-\u9fff]", text)
        terms = set(chars)
        terms |= set(a + b for a, b in zip(chars, chars[1:]))
        terms |= set(re.findall(r"[a-z0-9_]{2,}", text))
        terms |= set(re.findall(r"\d+(?:\.\d+)?", text))
        return terms

    @staticmethod
    def _similar(a: str, b: str) -> float:
        ta, tb = ShortTermMemory._terms(a), ShortTermMemory._terms(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    # ------------------------------------------------------------------ 降级链
    def build_block(self, query: str, budget_tokens: int) -> STMBlock:
        stages: List[Dict] = []
        dropped: List[int] = []
        recs = list(self.records)
        before_tokens = estimate_tokens(self.render(recs))
        stages.append({"stage": "raw", "count": len(recs), "tokens": before_tokens})

        # ---- 阶段 1：去重 ----
        recs, dup_dropped = self._dedupe(recs)
        dropped.extend(dup_dropped)
        stages.append({"stage": "dedupe", "count": len(recs),
                       "tokens": estimate_tokens(self.render(recs)), "dropped": len(dup_dropped)})
        if self.tracer and dup_dropped:
            self.tracer.event("stm.dedupe", merged=len(dup_dropped),
                              kept=len(recs), dropped_ids=dup_dropped)

        text = self.render(recs)
        # ---- 阶段 2：剔除低相关（只动非保护、且不在最近窗口内的） ----
        if estimate_tokens(text) > budget_tokens:
            recs, low_dropped = self._drop_low_relevance(recs, query)
            dropped.extend(low_dropped)
            text = self.render(recs)
            stages.append({"stage": "drop_low_rel", "count": len(recs),
                           "tokens": estimate_tokens(text), "dropped": len(low_dropped)})
            if self.tracer:
                self.tracer.event("stm.degrade", stage="drop_low_relevance",
                                  dropped=len(low_dropped), remaining=len(recs),
                                  tokens=estimate_tokens(text), budget=budget_tokens,
                                  dropped_ids=low_dropped[:20])

        # ---- 阶段 3：老片段压缩成摘要（保留决策 + 证据） ----
        if estimate_tokens(text) > budget_tokens and len(recs) > config.STM_KEEP_RECENT:
            recs, summary = self._compress_old(recs)
            text = self.render(recs)
            stages.append({"stage": "compress", "count": len(recs),
                           "tokens": estimate_tokens(text), "summary_lines": summary})
            if self.tracer:
                self.tracer.event("stm.degrade", stage="compress_summary",
                                  summary_lines=summary, remaining=len(recs),
                                  tokens=estimate_tokens(text), budget=budget_tokens)

        # ---- 阶段 4：兜底收缩（摘要行再压缩 → 只留最近窗口 + 保护记录） ----
        if estimate_tokens(text) > budget_tokens:
            recs = self._shrink(recs, budget_tokens)
            text = self.render(recs)
            stages.append({"stage": "shrink", "count": len(recs),
                           "tokens": estimate_tokens(text)})
            if self.tracer:
                self.tracer.event("stm.degrade", stage="shrink_keep_recent",
                                  remaining=len(recs), tokens=estimate_tokens(text),
                                  budget=budget_tokens)

        used = estimate_tokens(text)
        evidence = [f"#{r.id}:{r.meta.get('tool', r.kind)}" for r in recs
                    if r.protected and r.kind in {"observation", "action"}]
        block = STMBlock(text=text, used_tokens=used, before_tokens=before_tokens,
                         stages=stages, kept_evidence=evidence,
                         dropped_ids=dropped, rendered=len(recs))
        if self.tracer:
            self.tracer.event("stm.read", budget=budget_tokens, used=used,
                              before=before_tokens, records=len(recs),
                              stages=stages, evidence_ids=evidence[:10],
                              degraded=before_tokens > budget_tokens)
        return block

    def _dedupe(self, recs: List[Record]):
        kept: List[Record] = []
        dropped: List[int] = []
        for rec in recs:
            matched = None
            for prev in kept:
                if prev.kind != rec.kind:
                    continue
                if self._similar(prev.content, rec.content) >= config.STM_DEDUPE_SIM:
                    matched = prev
                    break
            if matched is None:
                kept.append(rec)
                continue
            # 重复的：如果是新的一条且对方没有保护、自己受保护，则替换位置保留新的
            if rec.protected and not matched.protected:
                kept[kept.index(matched)] = rec
                dropped.append(matched.id)
            else:
                matched.merged += 1
                dropped.append(rec.id)
        return kept, dropped

    def _drop_low_relevance(self, recs: List[Record], query: str):
        q_terms = self._terms(query)
        total = len(recs)
        keep_from = max(0, total - config.STM_KEEP_RECENT)
        recent_ids = {r.id for r in recs[keep_from:]}
        scored = []
        for i, rec in enumerate(recs):
            overlap = 0.0
            if q_terms:
                t = self._terms(rec.content)
                overlap = len(t & q_terms) / max(1, len(q_terms))
            recency = 1.0 / (1.0 + 0.15 * (total - 1 - i))
            score = 0.55 * overlap + 0.30 * recency + 0.15 * KIND_WEIGHT.get(rec.kind, 0.5)
            scored.append((score, rec))
        kept, dropped = [], []
        for score, rec in scored:
            if rec.protected or rec.id in recent_ids:
                kept.append(rec)
            elif score < config.STM_DROP_SCORE:
                dropped.append(rec.id)
            else:
                kept.append(rec)
        return kept, dropped

    def _summary_line(self, rec: Record) -> str:
        """压缩时每个片段只留一行，但必须保住"做了什么 + 结果 + 证据 id"。"""
        c = clip_chars(rec.content.replace("\n", " "), config.STM_LINE_MAX_CHARS)
        if rec.kind == "user":
            return f"用户要求：{c}"
        if rec.kind == "thought":
            return f"思考：{c}"
        if rec.kind == "action":
            tool = rec.meta.get("tool") or ""
            args = rec.meta.get("args")
            return f"动作：{tool} {clip_chars(str(args), 34) if args else ''}".strip()
        if rec.kind == "observation":
            ok = "成功" if rec.meta.get("success") else "失败"
            rid = rec.meta.get("row_id")
            return f"观察：{rec.meta.get('tool', '?')} {ok}" + (f"，涉及记录 id={rid}" if rid else "") + f"｜{c}"
        if rec.kind == "answer":
            return f"结论：{c}"
        return f"{rec.kind}：{c}"

    def _compress_old(self, recs: List[Record]):
        keep_from = max(0, len(recs) - config.STM_KEEP_RECENT)
        old, recent = recs[:keep_from], recs[keep_from:]
        # 保护记录不参与压缩：一旦并进摘要，它的 id 与 meta 就没了，
        # kept_evidence 会变空，"关键证据仍可追溯"这个承诺就落不了地。
        pinned = [r for r in old if r.protected]
        old = [r for r in old if not r.protected]
        if not old:
            return recs, 0
        lines = [self._summary_line(r) for r in old]
        total_lines = len(lines)
        lines = lines[:config.STM_COMPRESS_MAX_LINES]
        if total_lines > len(lines):
            lines.append(f"…另有 {total_lines - len(lines)} 条已折叠（关键动作与证据已保留在上面）")
        summary = Record(
            id=-1, kind="summary", content="\n".join(lines), protected=False,
            turn=old[-1].turn, step=old[-1].step,
            tokens=estimate_tokens("\n".join(lines)), ts="",
            meta={"compressed_ids": [r.id for r in old], "pinned": True},
        )
        return [summary] + pinned + recent, total_lines

    def _shrink(self, recs: List[Record], budget_tokens: int):
        # 先压摘要行本身，再只留最近窗口 + 保护记录
        candidate = []
        for rec in recs:
            if rec.kind == "summary":
                lines = rec.content.split("\n")
                keep = max(3, len(lines) // 2)
                content = "\n".join(lines[:keep]) + (f"\n…折叠 {len(lines) - keep} 行" if len(lines) > keep else "")
                candidate.append(Record(id=rec.id, kind=rec.kind, content=content,
                                        protected=False, turn=rec.turn, step=rec.step,
                                        tokens=estimate_tokens(content), ts=rec.ts, meta=rec.meta))
            else:
                candidate.append(rec)
        text = self.render(candidate)
        if estimate_tokens(text) <= budget_tokens:
            return candidate

        keep_from = max(0, len(candidate) - config.STM_KEEP_RECENT)
        trimmed = [r for i, r in enumerate(candidate)
                   if i >= keep_from or r.protected or r.kind == "summary"]
        text = self.render(trimmed)
        if estimate_tokens(text) <= budget_tokens:
            return trimmed

        # 最后手段：把非最近记录裁短（仍保留首行信息，不整条丢弃）
        final = []
        for i, r in enumerate(trimmed):
            if i >= len(trimmed) - config.STM_KEEP_RECENT or r.protected:
                final.append(r)
            else:
                short = clip_chars(r.content.replace("\n", " "), 48)
                final.append(Record(id=r.id, kind=r.kind, content=short, protected=r.protected,
                                    turn=r.turn, step=r.step, tokens=estimate_tokens(short),
                                    ts=r.ts, meta=r.meta, merged=r.merged))
        return final

    # ------------------------------------------------------------------ 摘要 / 统计
    def digest(self, max_items: int = 12) -> str:
        """给长期记忆用的会话摘要（任务结束时可写入长期记忆）。"""
        lines = [self._summary_line(r) for r in self.records if r.kind != "summary"]
        return "\n".join(lines[-max_items:])

    def stats(self) -> Dict:
        return {"records": len(self.records),
                "tokens": sum(r.tokens for r in self.records),
                "protected": sum(1 for r in self.records if r.protected)}
