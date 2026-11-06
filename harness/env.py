# -*- coding: utf-8 -*-
"""零依赖 .env 加载器：把 <项目根>/.env 里的键值读进 os.environ。

不引入 python-dotenv，只用标准库，理由与整个 harness 一致——零外部依赖即可跑。
规则：
  - 真实环境变量优先，.env 不会覆盖已存在的键（CI / 容器场景可完全绕过 .env）
  - 支持 # 注释、空行、KEY=VALUE、成对引号包裹
  - .env 不存在时静默返回 False，便于他人首次克隆后直接运行
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")


def load_dotenv(path: str = None) -> bool:
    """加载 .env 到 os.environ。返回是否读到了文件。"""
    path = path or ENV_PATH
    if not os.path.isfile(path):
        return False

    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    return True


def require(key: str, hint: str = "") -> str:
    """取必填环境变量，缺失时抛出可读错误，而不是等到调 API 才 401。"""
    value = os.getenv(key, "").strip()
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {key}。\n"
            f"  做法：复制 .env.example 为 .env，填入真实值。\n"
            + (f"  说明：{hint}" if hint else "")
        )
    return value
