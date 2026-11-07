# -*- coding: utf-8 -*-
"""30 轮长会话预算压测（离线，不调大模型、不连 MySQL）。

做法：模拟 30 轮真实形态的对话痕迹（protected 的用户输入 + Thought/Action/
Observation），每轮结束都按真实流程走一遍 ContextAssembler.build()，记录：
    - 每轮上下文总 token（必须全部 ≤ 3200 预算）
    - 降级触发轮数与各阶段出现次数（这就是简历里那个 N）
    - 第 1 轮的用户原话（protected）在第 30 轮的上下文里是否仍然存在（保护证据零丢失）

用法：./venv/Scripts/python.exe eval/eval_budget.py
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = Path(__file__).resolve().parent
RESULT_DIR = EVAL_DIR / "results"
os.environ["HARNESS_DATA_DIR"] = str(EVAL_DIR / "data")
os.environ["HARNESS_DB"] = str(EVAL_DIR / "data" / "eval.db")

from harness import config
from harness.context import ContextAssembler
from harness.short_term import ShortTermMemory
from harness.store import Store
from harness.task_state import TaskStateManager
from harness.trace import Tracer
import tools as T

ROUNDS = 30
OBS_TEMPLATE = ('{{"success": true, "count": 6, "data": ['
                '{{"id": {i}, "class_name": "高一(1)班", "student_no": "2024001",'
                ' "student_name": "张三", "chinese": 60.0, "math": 99.0, "english": 88.5}},'
                '{{"id": {j}, "class_name": "高一(1)班", "student_no": "2024003",'
                ' "student_name": "王五", "chinese": 85.0, "math": 92.0, "english": 93.0}}],'
                '"message": "查询成功，共 2 条（评测模拟观测）"}}')


def main():
    store = Store()                                   # eval/data/eval.db，与真实库隔离
    tracer = Tracer(store, "s-budget-eval", None, echo=False)
    stm = ShortTermMemory(store, "s-budget-eval", None, tracer)
    states = TaskStateManager(store, tracer)
    assembler = ContextAssembler(T.TOOL_DOCS, tracer)
    state, _ = states.load_or_create("t-budget-eval", "s-budget-eval", "预算压测：30 轮长会话")

    rows, stage_tally = [], {}
    for i in range(1, ROUNDS + 1):
        stm.begin_turn()
        stm.set_step(i)
        # 真实形态的痕迹：用户原话 protected；观测是最占空间的部分
        stm.add("user", f"第{i}轮用户提问：帮我查一下高一(1)班学生的成绩，"
                        f"把第 {i % 6 + 1} 号记录的数学加 1 分", protected=True)
        stm.add("thought", f"第{i}轮思路：先检索目标记录，确认存在后再执行更新。")
        stm.add("action", f'search_students {{"student_no": "202400{i % 6 + 1}"}}',
                meta={"tool": "search_students"})
        stm.add("observation", OBS_TEMPLATE.format(i=i, j=i % 6 + 3),
                meta={"tool": "search_students", "success": True, "row_id": i % 6 + 1})

        pkg = assembler.build(state=state, memories=[], stm=stm,
                              current=f"第{i}轮用户提问：帮我查一下高一(1)班学生的成绩",
                              query="高一(1)班 学生成绩 查询 更新")
        for s in pkg.stm_block.stages:
            if s["stage"] != "raw":
                stage_tally[s["stage"]] = stage_tally.get(s["stage"], 0) + 1
        rows.append({"round": i, "total_tokens": pkg.total_tokens,
                     "degraded": pkg.degraded, "stages": [s["stage"] for s in pkg.stm_block.stages]})

    joined_last = json.dumps([m["content"] for m in pkg.messages], ensure_ascii=False)
    evidence_kept = "第1轮用户提问" in joined_last
    over = [r for r in rows if r["total_tokens"] > config.TOKEN_BUDGET_TOTAL]

    print("=" * 62)
    print(f" 30 轮长会话预算压测（预算 {config.TOKEN_BUDGET_TOTAL} tok）")
    print(f"   首 5 轮 ：" + "  ".join(f"{r['total_tokens']}" for r in rows[:5]))
    print(f"   中段 15 轮：" + "  ".join(f"{r['total_tokens']}" for r in rows[14:19]))
    print(f"   末 5 轮 ：" + "  ".join(f"{r['total_tokens']}" for r in rows[-5:]))
    print(f"   峰值    ：{max(r['total_tokens'] for r in rows)} tok")
    print(f"   超预算轮数：{len(over)}（要求 0）")
    print(f"   触发降级轮数：{sum(1 for r in rows if r['degraded'])}")
    print(f"   各降级阶段出现次数：{stage_tally or '无'}")
    print(f"   第 1 轮用户原话在第 {ROUNDS} 轮上下文中仍存在：{'是' if evidence_kept else '否'}")
    print("=" * 62)

    RESULT_DIR.mkdir(exist_ok=True)
    out = RESULT_DIR / f"budget_{ROUNDS}r_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(
        {"summary": {"rounds": ROUNDS, "budget": config.TOKEN_BUDGET_TOTAL,
                     "peak_tokens": max(r["total_tokens"] for r in rows),
                     "over_budget_rounds": len(over),
                     "degraded_rounds": sum(1 for r in rows if r["degraded"]),
                     "stage_tally": stage_tally,
                     "round1_evidence_kept": evidence_kept},
         "rounds": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[保存] {out}")

    ok = not over and evidence_kept
    print(f"\n结论：{'通过 ✓' if ok else '未通过 ✗'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
