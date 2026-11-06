# -*- coding: utf-8 -*-
"""Harness 全局配置：预算、阈值、TTL、路径。所有数值都可以用环境变量覆盖。"""

import os

from .env import load_dotenv

load_dotenv()  # 先把 .env 读进环境变量；真实环境变量优先，不会被覆盖

# ---------- 路径 ----------
# BASE_DIR = 项目根目录（student-agent-react/）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("HARNESS_DATA_DIR", os.path.join(BASE_DIR, "data"))
LOG_DIR = os.getenv("HARNESS_LOG_DIR", os.path.join(BASE_DIR, "logs"))
DB_PATH = os.getenv("HARNESS_DB", os.path.join(DATA_DIR, "harness.db"))
TASK_DIR = os.path.join(DATA_DIR, "tasks")          # checkpoint 的 JSON 镜像
TRACE_JSONL_DIR = os.path.join(LOG_DIR, "trace")    # trace 的 JSONL 镜像

# ---------- 每轮上下文 token 预算 ----------
# 总预算刻意压得比模型窗口小很多：演示"预算驱动装配"，而不是把窗口塞满。
TOKEN_BUDGET_TOTAL = int(os.getenv("CTX_TOKEN_BUDGET", "3200"))
BUDGET = {
    "system": 950,        # 角色 + 工具定义 + 输出格式（固定，不参与裁剪）
    "task_state": 420,    # 结构化任务状态块
    "long_memory": 520,   # 长期记忆召回块
    "short_memory": 1080,  # 短期轨迹块（本轮回话 + Thought/Action/Observation）
    "current": 230,       # 当前指令
}

# ---------- 短期记忆降级参数 ----------
STM_KEEP_RECENT = 6          # 最近 N 条永不压缩（保证当前上下文连续）
STM_DEDUPE_SIM = 0.92        # 相似度 ≥ 此值视为重复
STM_DROP_SCORE = 0.22        # 相关度得分低于此值的非保护记录优先剔除
STM_COMPRESS_MAX_LINES = 12  # 摘要最多保留多少行
STM_LINE_MAX_CHARS = 64      # 摘要单行截断长度
STM_SUMMARY_KEEP = 1         # 压缩后最多保留几条摘要记录

# ---------- 长期记忆参数 ----------
LTM_TOP_K = int(os.getenv("LTM_TOP_K", "5"))
LTM_MIN_SIM = float(os.getenv("LTM_MIN_SIM", "0.22"))
LTM_SUPERSEDE_SIM = 0.80     # 与旧记忆高度相似但内容不同 → 旧记忆被标记 superseded（向量/文本骨架取大者）
LTM_DUPLICATE_SIM = 0.97     # 几乎同义 → 不重复写入
LTM_STALE_DAYS = 30          # 超过这个天数在召回时打 [可能过时] 标签
LTM_TTL_DAYS = {             # 新记忆默认有效期（天），None = 永久
    "preference": 365,
    "fact": 180,
    "conclusion": 30,
    "schema": 180,
}
LTM_KIND_WEIGHT = {"preference": 1.2, "fact": 1.1, "schema": 1.0, "conclusion": 0.9}

# ---------- 向量化 ----------
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")   # auto | openai | local
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_DIM = int(os.getenv("LOCAL_EMBED_DIM", "512"))

# ---------- ReAct 循环 ----------
REACT_MAX_STEPS = int(os.getenv("REACT_MAX_STEPS", "8"))
MAX_OBS_CHARS = 1500
STOP_SEQUENCES = ["\nObservation:", "Observation:"]

# ---------- 数据库（学生库，被 Agent 操作的那个库） ----------
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "school_db"),
    "charset": "utf8mb4",
}
SCHOOL_TABLE = "student_scores"


def ensure_dirs():
    for path in (DATA_DIR, LOG_DIR, TASK_DIR, TRACE_JSONL_DIR):
        os.makedirs(path, exist_ok=True)
