# -*- coding: utf-8 -*-
"""20 任务在线评测 runner。

用法：
    ./venv/Scripts/python.exe eval/eval_run.py            # 全量 20 条
    ./venv/Scripts/python.exe eval/eval_run.py T07 T18    # 只跑指定任务

隔离措施（都不碰真实数据）：
    - MySQL 用独立表 student_scores_eval（LIKE 真表克隆，跑完 DROP）
    - SQLite 记忆库 / checkpoint / trace 全部重定向到 eval/data/（HARNESS_* 环境变量）
    - 每次运行前清空 eval/data/ 的记忆库，避免上一轮的长期记忆被召回、模型不查库照抄
    - 独占锁 eval/.run.lock：两场评测并发会互相重置评测表、共用记忆库，数字就废了
    - 每条任务独立 session；T18-T20 三条记忆任务共用一个 session（跨任务记忆是测点）

产出：
    终端结果表 + eval/results/eval_tasks_<时间戳>.json
"""

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = Path(__file__).resolve().parent
RESULT_DIR = EVAL_DIR / "results"
os.environ["HARNESS_DATA_DIR"] = str(EVAL_DIR / "data")
os.environ["HARNESS_DB"] = str(EVAL_DIR / "data" / "eval.db")
os.environ["HARNESS_LOG_DIR"] = str(EVAL_DIR / "data" / "logs")   # trace 的 JSONL 也不落真实 logs/

from harness.env import load_dotenv
load_dotenv()

from harness import config
config.SCHOOL_TABLE = "student_scores_eval"      # 必须在 import tools / main 之前

import pymysql
from openai import OpenAI
import tools as T
import main as M
from eval_tasks import SEED_ROWS, TASKS

EVAL_TABLE = "student_scores_eval"
LOCK_PATH = EVAL_DIR / ".run.lock"


def acquire_lock():
    """挡住"两场评测同时跑"。

    两场会互相 DROP/CREATE 同一张评测表、共用同一个 SQLite 记忆库，
    结果既不可复现也不可信——实测踩到过：上一场写下的观测被下一场当成记忆直接照抄，
    出现"0 次工具调用却答对"的假通过。
    """
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"[中止] 已有一场评测在跑（{LOCK_PATH} 存在）。")
        print("       同时跑会互相重置评测表、共用记忆库，结果不可信。")
        print(f"       确认没有别的评测在跑后，删掉该文件再试：{LOCK_PATH}")
        sys.exit(1)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)


def release_lock():
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


class DB:
    """评测库查询助手：check 函数通过它核对数据库终态。"""

    def __init__(self):
        c = dict(config.DB_CONFIG)
        self.conn = pymysql.connect(host=c["host"], port=c.get("port", 3306),
                                    user=c["user"], password=c["password"],
                                    database=c["database"], charset="utf8mb4",
                                    autocommit=True,
                                    cursorclass=pymysql.cursors.DictCursor)

    def one(self, sql, params=None):
        with self.conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchone()

    def raw(self, student_no, field):
        """取原始值（不转型）：字符串列（class_name / student_name）必须走这个。"""
        row = self.one(f"SELECT {field} AS v FROM {EVAL_TABLE} WHERE student_no=%s", (student_no,))
        return row["v"] if row and row["v"] is not None else None

    def cell(self, student_no, field):
        """取数值列并转成 float。DDL 里 DECIMAL 会被驱动读成 Decimal，
        不做转型后续比较会出现 Decimal('99.00') != 99.0 的假失败。"""
        value = self.raw(student_no, field)
        return float(value) if value is not None else None

    def cell_id(self, student_no):
        row = self.one(f"SELECT id AS v FROM {EVAL_TABLE} WHERE student_no=%s", (student_no,))
        return int(row["v"]) if row else None

    def count(self, class_name):
        return int(self.one(f"SELECT COUNT(*) AS c FROM {EVAL_TABLE} WHERE class_name=%s",
                            (class_name,))["c"])

    def row_eq(self, student_no, **expect):
        row = self.one(f"SELECT * FROM {EVAL_TABLE} WHERE student_no=%s", (student_no,))
        if not row:
            return False
        for k, v in expect.items():
            got = row.get(k)
            if isinstance(v, float):
                if got is None or abs(float(got) - v) > 0.01:
                    return False
            elif got != v:
                return False
        return True

    def reset(self):
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {EVAL_TABLE}")
            cur.execute(f"CREATE TABLE {EVAL_TABLE} LIKE student_scores")
            cur.executemany(
                f"INSERT INTO {EVAL_TABLE} (class_name, student_no, student_name,"
                " chinese, math, english) VALUES (%s,%s,%s,%s,%s,%s)", SEED_ROWS)
        print(f"[准备] 评测表 {EVAL_TABLE} 已重建，{len(SEED_ROWS)} 行种子数据")

    def drop(self):
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {EVAL_TABLE}")
        self.conn.close()
        print(f"[清理] 评测表 {EVAL_TABLE} 已删除")


def make_agent(session_id: str):
    client = OpenAI(api_key=os.getenv("WORKBUDDY_API_KEY"),
                    base_url=os.getenv("WORKBUDDY_BASE_URL"),
                    timeout=60, max_retries=2)
    return M.MemoryReActAgent(client=client, model=os.getenv("WORKBUDDY_MODEL"),
                              session_id=session_id, verbose=False)


def run_task(task, db, mem_agent):
    """跑一条任务，返回结果 dict。mem_agent 非空表示沿用记忆会话。

    非记忆任务遇瞬时故障（API 503 / 响应为空）自动重试一次；记忆任务不重试——
    会话内的记忆已经写入，重试会让同一句话出现两份轨迹。
    """
    def attempt(agent):
        answers, steps, calls, tokens = [], 0, 0, 0
        t0 = time.time()
        try:
            for prompt in task["prompts"]:
                ans = agent.ask(prompt)
                answers.append(ans)
                steps += agent.last_steps
                calls += agent.last_tool_calls
                if agent.last_package:
                    tokens = max(tokens, agent.last_package.total_tokens)
            return answers, steps, calls, tokens, time.time() - t0, None
        except Exception as exc:
            return answers, steps, calls, tokens, time.time() - t0, f"{type(exc).__name__}: {exc}"

    answers, steps, calls, tokens, cost, error = attempt(
        mem_agent or make_agent(f"s-eval-{task['id']}"))
    if error and mem_agent is None:
        time.sleep(2)
        answers, steps, calls, tokens, cost, error = attempt(make_agent(f"s-eval-{task['id']}-r"))

    if error:
        ok, detail = False, f"执行异常：{error}"
    else:
        try:
            ok, detail = task["check"](answers, db)
        except Exception as exc:
            ok, detail = False, f"check 本身出错：{type(exc).__name__}: {exc}"
    return {"id": task["id"], "cat": task["cat"], "ok": bool(ok), "detail": str(detail)[:160],
            "steps": steps, "tool_calls": calls, "peak_tokens": tokens,
            "seconds": round(cost, 1),
            "answer_tail": (answers[-1] or "")[-100:]}


def reset_memory_store():
    """每次评测都从空记忆库起步。

    不清的话，上一轮写下的长期记忆会在这一轮被召回，模型可能不查库直接报出上次的结果：
    通过率既不可复现，也测不到它真会不会用工具。（T18-T20 的跨任务记忆仍然有效——
    它们共用的是同一个进程内的会话，不依赖磁盘上残留的旧库。）
    """
    db_file = Path(os.environ["HARNESS_DB"])
    for suffix in ("", "-wal", "-shm"):
        stale = Path(str(db_file) + suffix)
        if stale.exists():
            stale.unlink()
    task_dir = Path(os.environ["HARNESS_DATA_DIR"]) / "tasks"
    if task_dir.exists():
        shutil.rmtree(task_dir, ignore_errors=True)


def main():
    only = set(a for a in sys.argv[1:] if a.upper().startswith("T"))
    tasks = [t for t in TASKS if not only or t["id"] in {x.upper() for x in only}]

    reset_memory_store()
    db = DB()
    db.reset()
    mem_agent = None
    results = []
    try:
        for task in tasks:
            if task["cat"] == "记忆续用" and mem_agent is None:
                mem_agent = make_agent("s-eval-memory")
                print("[准备] 记忆任务共用会话 s-eval-memory（跨任务记忆是测点）")
            try:
                r = run_task(task, db, mem_agent if task["cat"] == "记忆续用" else None)
            except Exception as exc:              # 单条任务掀翻不了整场评测
                r = {"id": task["id"], "cat": task["cat"], "ok": False,
                     "detail": f"任务异常：{type(exc).__name__}: {exc}",
                     "steps": 0, "tool_calls": 0, "peak_tokens": 0,
                     "seconds": 0.0, "answer_tail": ""}
            mark = "✓" if r["ok"] else "✗"
            print(f"  [{mark}] {r['id']} {r['cat']}: {r['detail']}  "
                  f"({r['steps']}步/{r['tool_calls']}工具/{r['seconds']}s)")
            results.append(r)
    finally:
        db.drop()

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    by_cat = {}
    for r in results:
        c = by_cat.setdefault(r["cat"], [0, 0])
        c[0] += 1 if r["ok"] else 0
        c[1] += 1

    print("\n" + "=" * 62)
    print(f" 评测结果：{passed}/{total} 通过（{passed / total * 100:.0f}%）")
    for cat, (p, n) in by_cat.items():
        print(f"   {cat:6} {p}/{n}")
    print("=" * 62)

    RESULT_DIR.mkdir(exist_ok=True)
    out = RESULT_DIR / f"eval_tasks_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(
        {"summary": {"passed": passed, "total": total, "rate": round(passed / total, 3),
                     "by_category": {k: f"{v[0]}/{v[1]}" for k, v in by_cat.items()},
                     "model": os.getenv("WORKBUDDY_MODEL"),
                     "table": EVAL_TABLE, "date": datetime.now().isoformat()},
         "tasks": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[保存] {out}")


if __name__ == "__main__":
    acquire_lock()
    try:
        main()
    finally:
        release_lock()
