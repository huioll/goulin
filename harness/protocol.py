# -*- coding: utf-8 -*-
"""ReAct 文本协议解析：把模型输出解析成 Thought / Action / Action Input 或 Final Answer。

容忍代码块包裹、中文引号、模型自造 Observation 等脏输出，解析失败以 error 字段回传而非抛异常。
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

_RE_FINAL = re.compile(r"Final\s*Answer\s*[:：]", re.I)
_RE_THOUGHT = re.compile(
    r"Thought\s*[:：]\s*(.*?)(?=\n\s*(?:Action|Final\s*Answer)\s*[:：]|\Z)", re.S | re.I
)
_RE_ACTION = re.compile(r"Action\s*[:：]\s*([^\n]+)", re.I)
_RE_ACTION_INPUT = re.compile(
    r"Action\s*Input\s*[:：]\s*(.*?)(?=\n\s*(?:Observation|Thought|Action)\s*[:：]|\Z)", re.S | re.I
)
_RE_FINAL_BODY = re.compile(r"Final\s*Answer\s*[:：]\s*(.*)", re.S | re.I)


@dataclass
class ParsedStep:
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


def load_json_object(text: str) -> Dict[str, Any]:
    candidate = _strip_code_fence(text)
    try:
        data = json.loads(candidate)
    except Exception:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("Action Input 里找不到 JSON 对象")
        snippet = candidate[start:end + 1]
        snippet = snippet.replace("“", '"').replace("”", '"').replace("：", ":")
        data = json.loads(snippet)
    if not isinstance(data, dict):
        raise ValueError("Action Input 必须是 JSON 对象（{...}）")
    return data


def parse_react_output(raw: str) -> ParsedStep:
    step = ParsedStep(raw=raw or "")
    text = (raw or "").strip()
    if not text:
        step.error = "模型返回了空内容"
        return step

    match_action = _RE_ACTION.search(text)
    match_final = _RE_FINAL.search(text)
    use_final = match_final is not None and (
        match_action is None or match_final.start() < match_action.start()
    )

    thought_match = _RE_THOUGHT.search(text)
    if thought_match:
        step.thought = thought_match.group(1).strip()

    if use_final:
        body = _RE_FINAL_BODY.search(text)
        step.final = body.group(1).strip() if body else text[match_final.end():].strip()
        step.truncated_raw = text[:match_final.end()] + step.final
        return step

    if match_action is None:
        step.error = "没有解析到 Action 或 Final Answer"
        step.truncated_raw = text
        return step

    step.action = match_action.group(1).strip().strip("`'\"。.,， ")
    input_match = _RE_ACTION_INPUT.search(text, match_action.end())
    if not input_match:
        step.error = f"Action {step.action} 缺少 Action Input"
        step.truncated_raw = text[:match_action.end()]
        return step

    step.truncated_raw = text[:input_match.end()].strip()   # 丢掉模型多写的后续步骤
    try:
        step.action_input = load_json_object(input_match.group(1))
    except Exception as exc:
        step.error = f"Action Input 解析失败：{exc}"
    return step
