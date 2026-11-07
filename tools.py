# -*- coding: utf-8 -*-
"""数据库工具：4 个 CRUD + 1 个只读检索，全部参数化查询、不向上抛异常。

供 main.py 使用；写操作显式 commit、异常 rollback，错误统一降级为结构化 JSON。
"""

import inspect
import json
import time
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict

import pymysql

from harness import config

TABLE = config.SCHOOL_TABLE
DB_CONFIG = dict(config.DB_CONFIG, cursorclass=pymysql.cursors.DictCursor, autocommit=False)

WRITE_TOOLS = {"add_student", "update_student", "delete_student"}


@contextmanager
def get_conn():
    password = str(DB_CONFIG.get("password") or "")
    if not password or "替换" in password:
        raise RuntimeError("MySQL 密码未配置（还是占位符）：请改 harness/config.py 的 DB_CONFIG，"
                           "或设置环境变量 MYSQL_PASSWORD")
    conn = pymysql.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def _jsonable(row):
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


# --------------------------------------------------------------------------
# 工具实现
# --------------------------------------------------------------------------

def search_students(class_name=None, student_no=None, student_name=None, limit=10):
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
    class_name = _check_text(class_name, "class_name", "班级")
    student_no = _check_text(student_no, "student_no", "学号")
    student_name = _check_text(student_name, "student_name", "姓名")
    chinese = _check_score(chinese, "chinese")
    math = _check_score(math, "math")
    english = _check_score(english, "english")

    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id FROM {TABLE} WHERE class_name = %s AND student_no = %s LIMIT 1",
                    (class_name, student_no),
                )
                dup = cur.fetchone()
                if dup:
                    return _fail(f"新增失败：{class_name} 已存在学号 {student_no} 的记录"
                                 f"（id={dup['id']}），要修改请用 update_student")
                cur.execute(
                    f"INSERT INTO {TABLE} (class_name, student_no, student_name,"
                    f" chinese, math, english) VALUES (%s,%s,%s,%s,%s,%s)",
                    (class_name, student_no, student_name, chinese, math, english),
                )
                new_id = cur.lastrowid
            conn.commit()
            row = _fetch_by_id(conn, new_id)
        except Exception:
            conn.rollback()
            raise
    return _ok(message=f"新增成功，id={new_id}", data=row)


def get_student(id):
    student_id = _check_id(id)
    with get_conn() as conn:
        row = _fetch_by_id(conn, student_id)
    if row is None:
        return _fail(f"查询失败：不存在 id={student_id} 的记录")
    return _ok(data=row)


def update_student(id, class_name=None, student_no=None, student_name=None,
                   chinese=None, math=None, english=None):
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
                cur.execute(f"UPDATE {TABLE} SET {', '.join(fields)} WHERE id = %s",
                            (*params, student_id))
                affected = cur.rowcount
            conn.commit()
            after = _fetch_by_id(conn, student_id)
        except Exception:
            conn.rollback()
            raise

    message = ("更新已提交：字段值与原来一致，未发生变化（影响 0 行）" if affected == 0
               else f"更新成功（影响 {affected} 行）")
    return _ok(message=message, before=before, after=after)


def delete_student(id):
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

TOOL_DOCS = """### search_students
用途：按条件检索学生记录，返回列表（默认最多 10 条）。更新/删除前先用它拿到 id。
参数：class_name(可选,精确) / student_no(可选,精确) / student_name(可选,模糊) / limit(可选,默认10)

### get_student
用途：按主键 id 查询单条记录。
参数：id (integer, 必填)

### add_student
用途：新增一条学生成绩记录。同班级同学号重复会被拒绝。
参数：class_name(必填) / student_no(必填) / student_name(必填) /
      chinese, math, english (可选, 0~150, 默认0)

### update_student
用途：按 id 更新记录，只更新传入的字段。
参数：id(必填) + 以下任意组合：class_name / student_no / student_name / chinese / math / english

### delete_student
用途：按 id 删除记录，不可恢复。返回被删记录内容。
参数：id (integer, 必填)"""


def _clip(text: str, limit: int = config.MAX_OBS_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f" …(已截断，原长 {len(text)} 字符)"


def execute_action(name: str, arguments: Dict[str, Any]):
    """执行工具，返回 (Observation 字符串, 耗时毫秒, 结构化结果 dict)。绝不抛异常。"""
    handler = TOOLS.get(name)
    if handler is None:
        payload = json.dumps({"success": False,
                              "error": f"未知工具 {name}。可用工具：{', '.join(TOOLS)}"},
                             ensure_ascii=False)
        return payload, 0.0, {"success": False, "error": "unknown_tool"}

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
    return _clip(json.dumps(result, ensure_ascii=False, default=str)), cost_ms, result


# --------------------------------------------------------------------------
# 环境指纹 / 产物校验用的小工具（任务状态 checkpoint 会调用）
# --------------------------------------------------------------------------

def db_fingerprint() -> dict:
    """当前学生库的指纹：行数 + 最大 id + 表名。用于恢复时判断"现实是否变了"。"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS c, COALESCE(MAX(id),0) AS m FROM {TABLE}")
            row = cur.fetchone()
    return {"table": f"{DB_CONFIG['database']}.{TABLE}",
            "row_count": int(row["c"]), "max_id": int(row["m"])}


def fetch_row(row_id: int) -> dict:
    with get_conn() as conn:
        return _fetch_by_id(conn, row_id)


def row_fingerprint(row: dict) -> str:
    """记录内容的短哈希（忽略 created_at），用于比对"这条记录还是不是快照里那条"。"""
    if not row:
        return ""
    keys = ("id", "class_name", "student_no", "student_name", "chinese", "math", "english")
    payload = "|".join(f"{k}={row.get(k)}" for k in keys)
    import hashlib
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=4).hexdigest()


def make_artifact(row_id: int) -> str:
    """产物引用格式：student_scores#id=1@ab12cd34（表#主键@内容哈希）。"""
    row = fetch_row(row_id)
    if not row:
        return ""
    return f"{TABLE}#id={row_id}@{row_fingerprint(row)}"


def parse_artifact(ref: str):
    """解析成 (table, id, hash)；删除类产物可能没有 @hash。格式不合法返回 (None, None, None)。"""
    try:
        body, _, digest = ref.partition("@")
        table, key = body.split("#", 1)
        row_id = int(key.split("=", 1)[1])
        return table, row_id, digest
    except Exception:
        return None, None, None
