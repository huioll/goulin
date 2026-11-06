# -*- coding: utf-8 -*-
"""长期记忆：跨会话、向量化、有来源与时效。

写：只收"值得存"的东西（用户偏好、稳定事实、任务结论、库结构），带 kind / source / tags / TTL。
存：SQLite + 定长向量（BLOB），暴力余弦检索；换向量库只需替换本文件的 retrieve()。
查：用当前任务做 query 召回 top-k，过滤过期与被取代的记录，并给每条打上时间/来源/时效标签。
冲突：新写入的内容与旧记忆高度相似但不同 → 旧记录 superseded_by = 新 id，从此不再召回。
隔离：召回结果只作为"参考证据"注入，绝不写进指令区（见 context.py 的 LTM 区块声明）。
"""

import hashlib
import json
import re
import difflib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from . import config
from .embedding import cosine, from_blob, to_blob

_ID_IN_TEXT = re.compile(r"(?:id\s*[=＝:：]\s*|记录\s*#)(\d+)", re.I)
_NO_IN_TEXT = re.compile(r"\b(20\d{6,})\b")
_PREF_PATTERNS = [
    re.compile(r"(?:我(?:更)?(?:喜欢|习惯|偏好|希望|要求|需要)[^。；\n]{2,40})"),
    re.compile(r"(?:以后(?:都)?[^。；\n]{2,30})"),
    re.compile(r"(?:默认(?:用|是|按)[^。；\n]{2,30})"),
    re.compile(r"(?:记住[^。；\n]{2,50})"),
    re.compile(r"(?:我叫[^。；\n]{1,20})"),
]


def _skeleton(text: str) -> str:
    """把数字抹平后的"文本骨架"，用来判断两条记忆是不是在说同一件事。"""
    return re.sub(r"\d+(?:\.\d+)?", "#", re.sub(r"[^\w\u4e00-\u9fff]+", "", text or ""))


def skeleton_sim(a: str, b: str) -> float:
    sa, sb = _skeleton(a), _skeleton(b)
    if not sa or not sb:
        return 0.0
    return difflib.SequenceMatcher(None, sa, sb).ratio()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _age_days(ts: str) -> float:
    try:
        t = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
        return max(0.0, (datetime.now() - t).total_seconds() / 86400)
    except Exception:
        return 999.0


@dataclass
class MemoryHit:
    id: int
    text: str
    kind: str
    source: str
    created_at: str
    valid_until: Optional[str]
    tags: List[str]
    score: float
    sim: float
    stale: bool = False
    note: str = ""

    def label(self) -> str:
        age = int(_age_days(self.created_at))
        bits = [f"#{self.id}", self.kind, f"{age}天前", f"来源:{self.source}",
                f"sim={self.sim:.2f}", f"score={self.score:.2f}"]
        if self.stale:
            bits.append("可能过时")
        if self.note:
            bits.append(self.note)
        return "[" + "|".join(bits) + "]"


class LongTermMemory:
    def __init__(self, store, embedder, tracer=None):
        self.store = store
        self.embedder = embedder
        self.tracer = tracer

    # ------------------------------------------------------------------ 写
    def remember(self, text: str, kind: str = "fact", source: str = "agent",
                 task_id: str = None, session_id: str = None,
                 tags: List[str] = None, confidence: float = 0.8,
                 ttl_days: int = None) -> Dict:
        text = (text or "").strip()
        if len(text) < 4:
            return {"action": "skip", "reason": "内容太短"}
        if kind not in {"preference", "fact", "conclusion", "schema"}:
            kind = "fact"

        tags = list(dict.fromkeys(tags or []))
        vec = self.embedder.encode(text)
        dim = len(vec)
        content_hash = hashlib.blake2b(
            (kind + "|" + re.sub(r"\s+", "", text)).encode("utf-8"), digest_size=8
        ).hexdigest()

        candidates = self.store.ltm_all(dim=dim)
        best, best_sim = None, 0.0
        for row in candidates:
            if row.get("content_hash") == content_hash:
                self.store.ltm_touch(row["id"])
                if self.tracer:
                    self.tracer.event("ltm.skip", reason="duplicate", memory_id=row["id"],
                                      kind=kind, text=text[:80])
                return {"action": "skip", "id": row["id"], "reason": "内容完全重复"}
            # 向量相似度 + 文本骨架相似度取大者：向量抓语义，骨架抓"同一句话改了数值"
            sim = max(cosine(vec, from_blob(row["embedding"])), skeleton_sim(text, row["text"]))
            if sim > best_sim:
                best, best_sim = row, sim

        if best is not None and best_sim >= config.LTM_DUPLICATE_SIM:
            self.store.ltm_touch(best["id"])
            if self.tracer:
                self.tracer.event("ltm.skip", reason="near_duplicate", memory_id=best["id"],
                                  sim=round(best_sim, 4), kind=kind, text=text[:80])
            return {"action": "skip", "id": best["id"], "reason": f"近似重复（sim={best_sim:.2f}）"}

        ttl = ttl_days if ttl_days is not None else config.LTM_TTL_DAYS.get(kind)
        valid_until = None
        if ttl:
            from datetime import timedelta
            valid_until = (datetime.now() + timedelta(days=ttl)).strftime("%Y-%m-%d %H:%M:%S")

        new_id = self.store.ltm_insert(
            text=text, kind=kind, source=source, task_id=task_id, session_id=session_id,
            tags=tags, confidence=confidence, valid_until=valid_until,
            content_hash=content_hash, dim=dim, embedding=to_blob(vec),
        )

        superseded = None
        if best is not None and best_sim >= config.LTM_SUPERSEDE_SIM:
            self.store.ltm_supersede(best["id"], new_id)
            superseded = best["id"]
            if self.tracer:
                self.tracer.event("ltm.supersede", old_id=best["id"], new_id=new_id,
                                  sim=round(best_sim, 4), reason="新记忆与旧记忆高度相似但内容不同",
                                  old_text=best["text"][:80], new_text=text[:80])

        if self.tracer:
            self.tracer.event("ltm.write", memory_id=new_id, kind=kind, source=source,
                              tags=tags, ttl_days=ttl, valid_until=valid_until,
                              superseded=superseded, backend=self.embedder.backend,
                              dim=dim, text=text[:120])
        return {"action": "created", "id": new_id, "superseded": superseded,
                "valid_until": valid_until}

    # ------------------------------------------------------------------ 读
    def recall(self, query: str, top_k: int = None, min_sim: float = None,
               include_preferences: bool = True) -> List[MemoryHit]:
        top_k = top_k or config.LTM_TOP_K
        min_sim = config.LTM_MIN_SIM if min_sim is None else min_sim
        query = (query or "").strip()
        if not query:
            return []

        vec = self.embedder.encode(query)
        dim = len(vec)
        rows = self.store.ltm_all(dim=dim)
        if not rows:                       # provider 换了维度：退化为全部记录里挑同维度的
            rows = [r for r in self.store.ltm_all(dim=None) if r.get("dim") == dim]

        now = _now()
        scored: List[MemoryHit] = []
        expired = skipped = 0
        q_ids = set(_ID_IN_TEXT.findall(query))
        q_nos = set(_NO_IN_TEXT.findall(query))

        for row in rows:
            if row.get("superseded_by"):
                skipped += 1
                continue
            if row.get("valid_until") and row["valid_until"] < now:
                expired += 1
                continue
            sim = cosine(vec, from_blob(row["embedding"]))
            age = _age_days(row["created_at"])
            decay = 0.6 + 0.4 * pow(2.718281828, -age / 120.0)
            kind_w = config.LTM_KIND_WEIGHT.get(row["kind"], 1.0)
            note = ""
            if q_ids and set(_ID_IN_TEXT.findall(row["text"])) & q_ids:
                note = "与本轮同一记录，数值可能已被改过，须以数据库为准"
            elif q_nos and set(_NO_IN_TEXT.findall(row["text"])) & q_nos:
                note = "与本轮同一学号，请以数据库当前值为准"
            scored.append(MemoryHit(
                id=row["id"], text=row["text"], kind=row["kind"], source=row["source"],
                created_at=row["created_at"], valid_until=row.get("valid_until"),
                tags=json.loads(row.get("tags") or "[]"), score=sim * decay * kind_w,
                sim=sim, stale=age > config.LTM_STALE_DAYS, note=note,
            ))

        hits = [h for h in scored if h.sim >= min_sim]
        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:top_k]

        # 用户偏好是"全局约束"，与当前话题相似度低也该带上一条（否则改了话题就丢偏好）
        injected_pref = None
        if include_preferences and not any(h.kind == "preference" for h in hits):
            prefs = [h for h in scored if h.kind == "preference"]
            if prefs:
                best_pref = max(prefs, key=lambda h: (h.score, h.created_at))
                best_pref.note = (best_pref.note + "；" if best_pref.note else "") + "全局偏好"
                hits.append(best_pref)
                injected_pref = best_pref.id

        if self.tracer:
            self.tracer.event("ltm.recall", query=query[:100], top_k=top_k,
                              candidates=len(rows), hits=len(hits), expired=expired,
                              superseded=skipped, min_sim=min_sim,
                              preference_injected=injected_pref,
                              backend=self.embedder.backend,
                              result=[{"id": h.id, "kind": h.kind, "sim": round(h.sim, 3),
                                       "score": round(h.score, 3), "text": h.text[:60]}
                                      for h in hits])
        return hits

    # ------------------------------------------------------------------ 抽取
    def extract_and_store(self, user_input: str, answer: str, tool_events: List[Dict],
                          task_id: str, session_id: str) -> List[Dict]:
        """把有保存价值的内容落库。宁可漏存，也不要把每句话都塞进长期记忆。"""
        results = []

        # 1) 用户偏好 / 明确要求记住的事实
        for pattern in _PREF_PATTERNS:
            for match in pattern.finditer(user_input or ""):
                snippet = match.group(0).strip("，。； ")
                if len(snippet) >= 4:
                    results.append(self.remember(
                        snippet, kind="preference", source="user",
                        task_id=task_id, session_id=session_id,
                        tags=["偏好", "来自用户原话"], confidence=0.9,
                    ))
                    break

        # 2) 任务结论（发生过写入才存，带记录 id 作为证据）
        writes = [e for e in tool_events if e.get("tool") in
                  {"add_student", "update_student", "delete_student"} and e.get("ok")]
        if writes and answer:
            ids = [e.get("row_id") for e in writes if e.get("row_id")]
            text = (f"任务结论（{task_id}）：{'；'.join(e['summary'] for e in writes[:3])}。"
                    f"最终答复：{answer[:120]}")
            results.append(self.remember(
                text, kind="conclusion", source="agent", task_id=task_id,
                session_id=session_id, tags=["任务结论"] + [f"id={i}" for i in ids],
                confidence=0.75,
            ))

        # 3) 库结构类稳定事实（首次检索成功时记一条，便于跨会话复用）
        if any(e.get("tool") == "search_students" and e.get("ok") for e in tool_events) \
                and not self.store.ltm_all(dim=None):
            results.append(self.remember(
                f"学生库 {config.DB_CONFIG['database']}.{config.SCHOOL_TABLE} 的字段："
                f"id/class_name/student_no/student_name/chinese/math/english",
                kind="schema", source="tool", task_id=task_id, session_id=session_id,
                tags=["库结构"], confidence=0.95, ttl_days=None,
            ))
        return [r for r in results if r and r.get("action") != "skip"]

    # ------------------------------------------------------------------ 维护
    def stats(self) -> Dict:
        rows = self.store.ltm_all(dim=None, include_dead=True)
        by_kind: Dict[str, int] = {}
        for row in rows:
            by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
        return {"total": len(rows), "by_kind": by_kind,
                "superseded": sum(1 for r in rows if r.get("superseded_by")),
                "backend": self.embedder.backend, "dim": self.embedder.dim}
