# -*- coding: utf-8 -*-
"""
学生数据库 Agent —— 经典 ReAct 范式（Thought → Action → Observation 循环）

与同目录之外的「一次性 function call」版本的区别：
  * 旧版（student-agent-console/agent.py）：模型一次性返回 tool_calls，选定即执行，一轮结束（Plan-and-Execute）。
  * 本版：模型每轮只输出「一个 Thought + 一个 Action」，系统执行后把真实结果作为 Observation 喂回去，
    模型在下一轮重新判断——是先查再改、查不到就新增、还是直接结束。步骤不预先规划，允许中途改策略。
  * 通信协议不是 tool_calls，而是可读的文本 ReAct 协议，因此每一轮的思考过程都能打印出来看。

运行：python agent_react.py
依赖：pip install openai pymysql
"""

import inspect
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pymysql
from openai import OpenAI

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ==========================================================================
# 0. 配置区
# ==========================================================================

API_KEY = os.getenv("WORKBUDDY_API_KEY", "")
BASE_URL = os.getenv("WORKBUDDY_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("WORKBUDDY_MODEL", "deepseek-chat")

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "school_db"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "autocommit": False,
}

TABLE = "student_scores"
MAX_STEPS = int(os.getenv("REACT_MAX_STEPS", "8"))   # 单次提问最多循环几轮
MAX_OBS_CHARS = 1500                                 # 单条 Observation 截断长度，控制上下文
MAX_HISTORY_PAIRS = 4                                # 会话里保留几轮历史问答
TEMPERATURE = 0.0
STOP_SEQUENCES = ["\nObservation:", "Observation:"]   # 防止模型自己编造 Observation


# ==========================================================================
# 1. 数据库基座
# ==========================================================================

@contextmanager
def get_conn():
    conn = pymysql.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def _jsonable(row):
    """行 -> 可 JSON 序列化的 dict。"""
    if row is None:
        return None
    out = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            out[key] = float(value)
        elif isinstance(value, datetime):
            out[key] = value.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(value, date):
            out[key] = value.strftime("%Y-%m-%d")
        else:
            out[key] = value
    return out


def _fetch_by_id(conn, student_id):
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE} WHERE id = %s", (student_id,))
        return _jsonable(cur.fetchone())


def _ok(**kwargs):
    data = {"success": True}
    data.update(kwargs)
    return data


def _fail(message):
    return {"success": False, "error": message}


def _check_id(value):
    try:
        student_id = int(value)
    except (TypeError, ValueError):
        raise ValueError("id 必须是整数")
    if student_id <= 0:
        raise ValueError("id 必须是正整数")
    return student_id


def _check_text(value, field, label):
    if value is None or str(value).strip() == "":
        raise ValueError(f"{label}（{field}）不能为空")
    return str(value).strip()


def _check_score(value, field):
    if value is None:
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} 必须是数字")
    if score < 0 or score > 150:
        raise ValueError(f"{field} 必须在 0~150 之间，收到 {score}")
    return round(score, 2)


# ==========================================================================
# 2. 数据库工具（4 个 CRUD + 1 个检索）
#    ReAct 要能"先查后改"，所以必须有按班级/学号/姓名检索的只读工具；
#    另外 4 个与旧项目完全一致。
# ==========================================================================

def search_students(class_name=None, student_no=None, student_name=None, limit=10):
    """按条件检索学生记录，返回列表（ReAct 的第一步通常是它）。"""
    try:
        limit = int(limit) if limit is not None else 10
    except (TypeError, ValueError):
        raise ValueError("limit 必须是整数")
    limit = max(1, min(limit, 50))

    where, params = [], []
    if class_name is not None and str(class_name).strip():
        where.append("class_name = %s")
        params.append(str(class_name).strip())
    if student_no is not None and str(student_no).strip():
        where.append("student_no = %s")
        params.append(str(student_no).strip())
    if student_name is not None and str(student_name).strip():
        where.append("student_name LIKE %s")
        params.append(f"%{str(student_name).strip()}%")

    sql = f"SELECT * FROM {TABLE}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id ASC LIMIT %s"
    params.append(limit)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = [_jsonable(r) for r in cur.fetchall()]

    if not rows:
        return _ok(count=0, data=[], message="没有匹配的记录")
    return _ok(count=len(rows), data=rows)


def add_student(class_name, student_no, student_name,
                chinese=None, math=None, english=None):
    """新增学生成绩记录。"""
    class_name = _check_text(class_name, "class_name", "班级")
    student_no = _check_text(student_no, "student_no", "学号")
    student_name = _check_text(student_name, "student_name", "姓名")
    scores = {
        "chinese": _check_score(chinese, "chinese"),
        "math": _check_score(math, "math"),
        "english": _check_score(english, "english"),
    }

    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id FROM {TABLE} WHERE class_name = %s AND student_no = %s LIMIT 1",
                    (class_name, student_no),
                )
                dup = cur.fetchone()
                if dup:
                    return _fail(
                        f"新增失败：{class_name} 已存在学号 {student_no} 的记录（id={dup['id']}），"
                        f"要修改请用 update_student"
                    )
                cur.execute(
                    f"INSERT INTO {TABLE} "
                    f"(class_name, student_no, student_name, chinese, math, english) "
                    f"VALUES (%s, %s, %s, %s, %s, %s)",
                    (class_name, student_no, student_name,
                     scores["chinese"], scores["math"], scores["english"]),
                )
                new_id = cur.lastrowid
            conn.commit()
            row = _fetch_by_id(conn, new_id)
        except Exception:
            conn.rollback()
            raise

    return _ok(message=f"新增成功，id={new_id}", data=row)


def get_student(id):
    """按 id 查单条记录。"""
    student_id = _check_id(id)
    with get_conn() as conn:
        row = _fetch_by_id(conn, student_id)
    if row is None:
        return _fail(f"查询失败：不存在 id={student_id} 的记录")
    return _ok(data=row)


def update_student(id, class_name=None, student_no=None, student_name=None,
                   chinese=None, math=None, english=None):
    """按 id 更新，只更新传入的字段。"""
    student_id = _check_id(id)
    fields, params = [], []
    if class_name is not None:
        fields.append("class_name = %s")
        params.append(_check_text(class_name, "class_name", "班级"))
    if student_no is not None:
        fields.append("student_no = %s")
        params.append(_check_text(student_no, "student_no", "学号"))
    if student_name is not None:
        fields.append("student_name = %s")
        params.append(_check_text(student_name, "student_name", "姓名"))
    if chinese is not None:
        fields.append("chinese = %s")
        params.append(_check_score(chinese, "chinese"))
    if math is not None:
        fields.append("math = %s")
        params.append(_check_score(math, "math"))
    if english is not None:
        fields.append("english = %s")
        params.append(_check_score(english, "english"))

    if not fields:
        return _fail("更新失败：没有提供任何需要修改的字段（至少传一个字段）")

    with get_conn() as conn:
        try:
            before = _fetch_by_id(conn, student_id)
            if before is None:
                return _fail(f"更新失败：不存在 id={student_id} 的记录")
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {TABLE} SET {', '.join(fields)} WHERE id = %s",
                    (*params, student_id),
                )
                affected = cur.rowcount
            conn.commit()
            after = _fetch_by_id(conn, student_id)
        except Exception:
            conn.rollback()
            raise

    if affected == 0:
        # MySQL 在"提交的值与原来完全一致"时 rowcount 也是 0，这不是失败，要说清楚
        message = "更新已提交：字段值与原来一致，未发生变化（影响 0 行）"
    else:
        message = f"更新成功（影响 {affected} 行）"
    return _ok(message=message, before=before, after=after)


def delete_student(id):
    """按 id 删除记录。"""
    student_id = _check_id(id)
    with get_conn() as conn:
        try:
            before = _fetch_by_id(conn, student_id)
            if before is None:
                return _fail(f"删除失败：不存在 id={student_id} 的记录")
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {TABLE} WHERE id = %s", (student_id,))
                affected = cur.rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return _ok(message=f"删除成功（{affected} 行）", deleted=before)


TOOLS = {
    "search_students": search_students,
    "add_student": add_student,
    "get_student": get_student,
    "update_student": update_student,
    "delete_student": delete_student,
}


# ==========================================================================
# 3. 工具说明（写进提示词，让模型知道 Action 有哪些）
# ==========================================================================

TOOL_DOCS = """### search_students
用途：按条件检索学生记录，返回列表（默认最多 10 条，最多 50 条）。更新/删除前先用它拿到 id。
参数：
- class_name  (string, 可选) 班级，精确匹配，如 "高三(1)班"
- student_no  (string, 可选) 学号，精确匹配，如 "2026002"
- student_name(string, 可选) 姓名，模糊匹配，如 "小红"
- limit       (integer,可选) 返回条数上限，默认 10
说明：三个条件都不传 = 列出前 N 条记录。

### get_student
用途：按主键 id 查询单条记录。
参数：
- id (integer, 必填) 学生记录主键

### add_student
用途：新增一条学生成绩记录。同班级同学号重复会被拒绝。
参数：
- class_name   (string, 必填) 班级
- student_no   (string, 必填) 学号
- student_name (string, 必填) 姓名
- chinese      (number, 可选) 语文分数 0~150，不传默认 0
- math         (number, 可选) 数学分数 0~150，不传默认 0
- english      (number, 可选) 英语分数 0~150，不传默认 0

### update_student
用途：按 id 更新记录，只更新传入的字段，未传字段保持不变。
参数：
- id (integer, 必填) 学生记录主键
- class_name / student_no / student_name / chinese / math / english (均可选)

### delete_student
用途：按 id 删除记录，不可恢复。返回被删记录内容。
参数：
- id (integer, 必填) 学生记录主键"""


SYSTEM_PROMPT = f"""你是一个用 ReAct（Thought → Action → Observation）循环操作 MySQL 学生成绩库的助手。
数据库：school_db.student_scores（字段：id, class_name 班级, student_no 学号, student_name 姓名,
chinese 语文, math 数学, english 英语, created_at）。你无法直接访问数据库，只能通过下面的工具。

## 可用工具
{TOOL_DOCS}

## 输出格式（每一轮都必须严格照做）
Thought: 用一两句话说清：现在已经知道什么、还缺什么、这一步为什么调这个工具。
Action: 工具名（必须是上面 5 个之一）
Action Input: 一个严格的单行 JSON 对象，键名用工具参数名

之后立刻停止输出，系统会执行这个工具并回给你一行：
Observation: <工具的真实返回 JSON>

当信息足够、不需要再调工具时，输出：
Thought: 为什么已经可以回答用户
Final Answer: 面向用户的中文答复

## 例子 1（先查后改）
问题：把学号 2026002 的小红的数学改成 99
Thought: 我没有这条记录的 id，先用学号检索拿到 id。
Action: search_students
Action Input: {{"student_no": "2026002"}}
Observation: {{"success": true, "count": 1, "data": [{{"id": 7, "student_name": "小红", "math": 88.0}}]}}
Thought: 找到了 id=7 的小红，只需改数学，用 update_student 只传 id 和 math。
Action: update_student
Action Input: {{"id": 7, "math": 99}}
Observation: {{"success": true, "message": "更新成功（影响 1 行）"}}
Thought: 更新已完成，可以回答用户。
Final Answer: 已把小红（id=7）的数学改为 99 分。

## 例子 2（查不到就新增）
问题：高三(1)班学号 2026009 的小明如果还没有成绩就帮他录入，语文 90 数学 88 英语 86
Thought: 先按班级和学号查一下有没有记录。
Action: search_students
Action Input: {{"class_name": "高三(1)班", "student_no": "2026009"}}
Observation: {{"success": true, "count": 0, "data": [], "message": "没有匹配的记录"}}
Thought: 确实没有，用户给了班级、学号、姓名和三科分数，直接新增。
Action: add_student
Action Input: {{"class_name": "高三(1)班", "student_no": "2026009", "student_name": "小明", "chinese": 90, "math": 88, "english": 86}}
Observation: {{"success": true, "message": "新增成功，id=12", "data": {{"id": 12}}}}
Thought: 新增成功，可以回答用户。
Final Answer: 原来没有 2026009 的记录，已新增（id=12）：小明，语文 90 / 数学 88 / 英语 86。

## 硬性规则
1. 每次回复只能有「一个 Thought + 一个 Action + 一个 Action Input」，然后立刻停；绝不要自己编造 Observation，也不要在一轮里写两个 Action。
2. 任何需要 id 的操作，先 search_students 查出来再动；禁止猜 id。
3. 删除/更新前必须确认目标记录（学号、姓名对得上），对不上就再来一轮查询或向用户确认。
4. 条件型任务（"有就改、没有就加"）要真的分轮：先查 → 看 Observation → 再决定改还是加，不要一次做完。
5. 工具返回 success=false 时，先读 error 想补救办法（改参数、换工具、或直接告诉用户），不要用完全相同的参数重复调用。
6. 信息不全（如缺班级、学号、姓名）不要编造，直接 Final Answer 追问。
7. 最终答复用中文、简洁；涉及记录时给出 id / 班级 / 学号 / 姓名 / 三科分数，不要把 JSON 原文贴给用户。"""


# ==========================================================================
# 4. ReAct 文本协议解析
# ==========================================================================

# 兼容中英文冒号
_RE_FINAL = re.compile(r"Final\s*Answer\s*[:：]", re.I)
_RE_THOUGHT = re.compile(r"Thought\s*[:：]\s*(.*?)(?=\n\s*(?:Action|Final\s*Answer)\s*[:：]|\Z)", re.S | re.I)
_RE_ACTION = re.compile(r"Action\s*[:：]\s*([^\n]+)", re.I)
_RE_ACTION_INPUT = re.compile(
    r"Action\s*Input\s*[:：]\s*(.*?)(?=\n\s*(?:Observation|Thought|Action)\s*[:：]|\Z)", re.S | re.I
)
_RE_FINAL_BODY = re.compile(r"Final\s*Answer\s*[:：]\s*(.*)", re.S | re.I)


@dataclass
class ParsedStep:
    """一次模型输出的解析结果。"""
    thought: str = ""
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    final: Optional[str] = None
    error: Optional[str] = None
    raw: str = ""
    truncated_raw: str = ""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.S)
    return fence.group(1).strip() if fence else text


def _load_json_object(text: str) -> Dict[str, Any]:
    """从 Action Input 文本里抠出 JSON 对象，容忍代码块、前后废话、中文引号。"""
    candidate = _strip_code_fence(text)
    try:
        data = json.loads(candidate)
    except Exception:
        # 退一步：抓第一个 {...} 片段
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("Action Input 里找不到 JSON 对象")
        snippet = candidate[start:end + 1]
        snippet = snippet.replace("“", '"').replace("”", '"').replace("：", ":")
        data = json.loads(snippet)
    if not isinstance(data, dict):
        raise ValueError("Action Input 必须是 JSON 对象（{...}），不能是数组或字符串")
    return data


def parse_react_output(raw: str) -> ParsedStep:
    """把模型这一轮的文本解析成 Thought / Action / Action Input 或 Final Answer。"""
    step = ParsedStep(raw=raw)
    text = raw.strip()
    if not text:
        step.error = "模型返回了空内容"
        step.truncated_raw = text
        return step

    # 只看第一个 Action 之前的部分，避免模型多写了一堆后续步骤
    match_action = _RE_ACTION.search(text)
    match_final = _RE_FINAL.search(text)
    use_final = match_final is not None and (match_action is None or match_final.start() < match_action.start())

    thought_match = _RE_THOUGHT.search(text)
    if thought_match:
        step.thought = thought_match.group(1).strip()

    if use_final:
        body = _RE_FINAL_BODY.search(text)
        step.final = (body.group(1).strip() if body else text[match_final.end():].strip())
        step.truncated_raw = text[:match_final.end()] + step.final
        return step

    if match_action is None:
        step.error = "没有解析到 Action 或 Final Answer"
        step.truncated_raw = text
        return step

    action_name = match_action.group(1).strip().strip("`'\"。.,， ")
    step.action = action_name

    input_match = _RE_ACTION_INPUT.search(text, match_action.end())
    if not input_match:
        step.error = f"Action {action_name} 缺少 Action Input"
        step.truncated_raw = text[:match_action.end()]
        return step

    raw_input = input_match.group(1)
    step.truncated_raw = text[:input_match.end()].strip()   # 丢掉模型多写的 Observation / 下一步

    try:
        step.action_input = _load_json_object(raw_input)
    except Exception as exc:
        step.error = f"Action Input 解析失败：{exc}"
        step.action_input = None
    return step


# ==========================================================================
# 5. 执行 Action，生成 Observation
# ==========================================================================

def _clip(text: str, limit: int = MAX_OBS_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f" …(已截断，原长 {len(text)} 字符)"


def push_user(messages: List[Dict[str, str]], content: str):
    """追加一条 user 消息；若上一条已经是 user（例如强制收尾提示），则合并成一条。"""
    if messages and messages[-1].get("role") == "user":
        messages[-1] = {"role": "user", "content": messages[-1]["content"] + "\n\n" + content}
    else:
        messages.append({"role": "user", "content": content})


def execute_action(name: str, arguments: Dict[str, Any]):
    """真正执行工具，返回 (给模型看的 Observation 字符串, 耗时毫秒)。绝不抛异常。"""
    handler = TOOLS.get(name)
    if handler is None:
        payload = json.dumps({
            "success": False,
            "error": f"未知工具 {name}。可用工具：{', '.join(TOOLS)}",
        }, ensure_ascii=False)
        return payload, 0.0

    allowed = set(inspect.signature(handler).parameters)
    extra = [k for k in arguments if k not in allowed]
    clean = {k: v for k, v in arguments.items() if k in allowed}

    started = time.perf_counter()
    try:
        result = handler(**clean)
    except TypeError as exc:
        result = {"success": False, "error": f"参数不匹配：{exc}（收到参数 {list(arguments)}）"}
    except ValueError as exc:
        result = {"success": False, "error": f"参数校验失败：{exc}"}
    except pymysql.MySQLError as exc:
        result = {"success": False, "error": f"数据库错误：{exc}"}
    except Exception as exc:
        result = {"success": False, "error": f"执行失败：{type(exc).__name__}: {exc}"}
    cost_ms = (time.perf_counter() - started) * 1000

    if extra:
        result["ignored_params"] = extra
    payload = json.dumps(result, ensure_ascii=False, default=str)
    return _clip(payload), cost_ms


# ==========================================================================
# 6. 打印（让 Thought / Action / Observation 一眼能看清）
# ==========================================================================

RULE = "-" * 74


def _indent(text: str, prefix: str = " " * 15) -> str:
    lines = str(text).splitlines() or [""]
    return ("\n" + prefix).join(lines)


def log_thought(text: str):
    print(f"  Thought     : {_indent(text)}")


def log_action(name: str, arguments: Any):
    print(f"  Action      : {name}")
    print(f"  Action Input: {json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else arguments}")


def log_observation(obs: str, cost_ms: Optional[float] = None):
    suffix = f"    ({cost_ms:.1f} ms)" if cost_ms is not None else ""
    print(f"  Observation : {_indent(obs, ' ' * 16)}{suffix}")


# ==========================================================================
# 7. ReAct Agent 主体
# ==========================================================================

class ReactStudentAgent:
    """把「模型输出 -> 解析 -> 执行 -> 回填 Observation」跑成循环。"""

    def __init__(self, client, model: str = MODEL, max_steps: int = MAX_STEPS, verbose: bool = True):
        self.client = client
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.history: List[Dict[str, str]] = []   # 只保留历次最终问答，供追问使用
        self.last_steps = 0
        self.last_tool_calls = 0

    # ---------- 模型调用 ----------
    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        kwargs = dict(model=self.model, messages=messages,
                      temperature=TEMPERATURE, stop=STOP_SEQUENCES)
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "stop" in str(exc).lower():      # 少数网关不支持 stop，降级重试一次
                kwargs.pop("stop")
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        if not resp.choices:
            raise RuntimeError("模型没有返回任何 choices")
        return (resp.choices[0].message.content or "").strip()

    # ---------- 主循环 ----------
    def run(self, question: str) -> str:
        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.history[-(MAX_HISTORY_PAIRS * 2):])
        messages.append({"role": "user", "content": question})

        self.last_steps = 0
        self.last_tool_calls = 0
        last_signature: Optional[str] = None
        repeat_count = 0

        for step in range(1, self.max_steps + 1):
            self.last_steps = step
            if self.verbose:
                print(f"\n{RULE}\n[第 {step} 轮] 调用模型，让它基于现有 Observation 思考")
                print(RULE)

            raw = self._call_model(messages)
            parsed = parse_react_output(raw)

            if self.verbose and parsed.thought:
                log_thought(parsed.thought)

            # ① 已经给出最终答案：结束循环
            if parsed.final is not None:
                if self.verbose:
                    print("  (模型判断信息已足够，不再调用工具 → 终止循环)")
                self._remember(question, parsed.final)
                return parsed.final

            # ② 解析失败：把错误当成 Observation 喂回去，让模型自我纠正（ReAct 的自恢复能力）
            if parsed.error and parsed.action_input is None:
                if self.verbose:
                    if parsed.action:
                        log_action(parsed.action, "（缺失或不是合法 JSON）")
                    print(f"  [!] 解析失败：{parsed.error}")
                    print(f"  [!] 原始输出：{_indent(parsed.raw[:400], ' ' * 16)}")
                observation = json.dumps({
                    "success": False,
                    "error": f"{parsed.error}。请严格按格式重来：Thought / Action / Action Input(单行 JSON)，"
                             f"或直接给 Final Answer。",
                }, ensure_ascii=False)
                messages.append({"role": "assistant", "content": parsed.truncated_raw or raw})
                push_user(messages, f"Observation: {observation}")
                if self.verbose:
                    log_observation(observation)
                continue

            # ③ 正常 Action：执行工具，拿真实 Observation
            action_name = parsed.action or ""
            arguments = parsed.action_input or {}
            signature = f"{action_name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
            if signature == last_signature:
                repeat_count += 1
            else:
                repeat_count = 0
            last_signature = signature

            if repeat_count >= 2:
                observation = json.dumps({
                    "success": False,
                    "error": "检测到你用完全相同的参数连续调用同一个工具，已阻止。请换参数、换工具，或直接给 Final Answer。",
                }, ensure_ascii=False)
                cost = None
            else:
                observation, cost = execute_action(action_name, arguments)
                self.last_tool_calls += 1

            if self.verbose:
                log_action(action_name, arguments)
                log_observation(observation, cost)

            messages.append({"role": "assistant", "content": parsed.truncated_raw or raw})
            push_user(messages, f"Observation: {observation}")

        # ④ 步数用尽：强制收尾，不再允许调工具
        if self.verbose:
            print(f"\n{RULE}\n[达到最大步数 {self.max_steps}] 强制要求模型基于已有 Observation 作答\n{RULE}")
        push_user(
            messages,
            "（已达到最大循环步数，请不要再调用工具。直接输出 Thought 和 Final Answer，"
            "并说明哪些部分没来得及完成。）",
        )
        raw = self._call_model(messages)
        parsed = parse_react_output(raw)
        final = parsed.final or raw.strip()
        self._remember(question, final)
        return final

    def _remember(self, question: str, answer: str):
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        keep = MAX_HISTORY_PAIRS * 2
        if len(self.history) > keep:
            self.history = self.history[-keep:]


# ==========================================================================
# 8. 启动自检 + 主循环
# ==========================================================================

def check_llm_config() -> List[str]:
    problems = []
    if not API_KEY or "替换" in API_KEY:
        problems.append("API Key 未配置：改 API_KEY，或设置环境变量 WORKBUDDY_API_KEY")
    if not BASE_URL or "your-workbuddy-host" in BASE_URL:
        problems.append("接口地址未配置：改 BASE_URL（OpenAI 兼容地址，一般以 /v1 结尾）")
    if not MODEL or "your-model-name" in MODEL:
        problems.append("模型名未配置：改 MODEL 为接口支持的模型名")
    return problems


def check_db():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.execute("SHOW TABLES LIKE %s", (TABLE,))
                if cur.fetchone() is None:
                    print(f"[警告] {DB_CONFIG['database']} 里没有表 {TABLE}，"
                          f"请先执行 ../student-agent-console/sql/init_student_scores.sql")
                    return
                cur.execute(f"SELECT COUNT(*) AS c FROM {TABLE}")
                total = cur.fetchone()["c"]
        print(f"[信息] 数据库连接正常：{DB_CONFIG['database']}.{TABLE} 共 {total} 条记录")
    except Exception as exc:
        print(f"[警告] 数据库连接失败：{exc}")
        print("       请检查 MYSQL_* 配置或确认 MySQL 已启动。可继续对话，但工具会返回错误。")


def main():
    print("=" * 74)
    print(" 学生数据库 Agent · ReAct 版（Thought → Action → Observation 循环）")
    print("=" * 74)

    problems = check_llm_config()
    if problems:
        for p in problems:
            print(f"[配置错误] {p}")
        print("\n请先在 agent_react.py 顶部配置区填写，或用环境变量传入。")
        sys.exit(1)

    print(f"[信息] 模型：{MODEL}    接口：{BASE_URL}")
    print(f"[信息] 最大循环步数：{MAX_STEPS}（改环境变量 REACT_MAX_STEPS 可调）")
    check_db()

    print("\n这个版本每一轮都会打印：Thought（模型想什么）→ Action（调哪个工具）"
          "→ Observation（数据库真实返回）")
    print("\n试这些指令（能明显看到多轮循环）：")
    print("  把学号2024001的数学改成99")
    print("  2026009 如果还没有记录就录入，语文90 数学88 英语86")
    print("  查一下高一(1)班有几个人，把最低分的那个英语加5分")
    print("  删除学号2024003的学生（会先查再删）")
    print("\n命令：exit / 退出 结束    clear / 清空 重置对话历史")
    print("-" * 74)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60, max_retries=2)
    agent = ReactStudentAgent(client=client, model=MODEL, max_steps=MAX_STEPS, verbose=True)

    while True:
        try:
            question = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q", "bye"} or question in {"退出", "再见"}:
            print("已退出。")
            break
        if question.lower() in {"clear", "reset"} or question in {"清空", "重置"}:
            agent.history.clear()
            print("已清空对话历史。")
            continue

        try:
            answer = agent.run(question)
        except KeyboardInterrupt:
            print("\n已取消本轮。")
            continue
        except Exception as exc:
            print(f"\n[错误] 调用失败：{type(exc).__name__}: {exc}")
            print("       请检查网络、BASE_URL、MODEL 是否有效。")
            continue

        print(f"\n{RULE}\n[最终回答]\n{answer}\n{RULE}")
        print(f"[统计] 共 {agent.last_steps} 轮思考，{agent.last_tool_calls} 次工具调用")


if __name__ == "__main__":
    main()
