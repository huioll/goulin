# -*- coding: utf-8 -*-
"""极简 token 估算。

不引入 tiktoken（避免额外依赖和下载模型）：中文按 1 字 ≈ 1 token，其他字符按 4 字符 ≈ 1 token 估。
对"预算控制"这个用途足够——宁可高估，也不要撑爆窗口。
"""

import re

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def estimate_tokens(text) -> int:
    if not text:
        return 0
    text = str(text)
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return int(cjk + other / 3.5) + 1


def clip_chars(text: str, max_chars: int, tail: str = "…") -> str:
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - len(tail))] + tail


def clip_tokens(text: str, max_tokens: int) -> str:
    """按估算 token 粗暴裁剪（只用于展示层的兜底，不用于短期记忆的压缩）。"""
    if estimate_tokens(text) <= max_tokens:
        return text
    # 中文密集时 token≈字数，先按字符裁再校验
    out = clip_chars(text, max(20, int(max_tokens * 1.1)))
    while estimate_tokens(out) > max_tokens and len(out) > 20:
        out = clip_chars(out, int(len(out) * 0.85))
    return out
