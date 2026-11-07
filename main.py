# -*- coding: utf-8 -*-
"""带三层记忆的 ReAct Agent（Harness 驱动）。

每一轮的流程（记忆全部在外层，模型本身不存任何状态）：

    读取 checkpoint → 核对现场（恢复时） → 召回长期记忆
        → 按预算装配上下文（系统 + 任务状态 + 长期记忆 + 短期轨迹）
        → 模型输出 Thought/Action
        → 执行工具 → Observation 写进短期记忆
        → 更新任务状态（动作、产物、未确认问题）并落 checkpoint
        → 回到装配，直到模型给 Final Answer

运行：
    python main.py              # 交互
    python main.py --trace      # 交互 + 打印每条 trace
    python main.py --task t-xxx # 恢复某个任务（会先做恢复校验）
    python main.py --show t-xxx # 看某个任务的 checkpoint
    python main.py --stats      # 看长期记忆/短期记忆/任务统计
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from openai import OpenAI

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from harness import config
from harness.context import ContextAssembler
from harness.embedding import Embedder
from harness.long_term import LongTermMemory
from harness.protocol import parse_react_output
from harness.short_term import ShortTermMemory
from harness.store import Store
from harness.task_state import STATUS_DONE, STATUS_RUNNING, TaskStateManager
from harness.tokens import clip_chars, estimate_tokens
from harness.trace import Tracer
import tools as T

API_KEY = os.getenv("WORKBUDDY_API_KEY", "sk-6fc1c3abef954df59420c0256d3c5b02")
BASE_URL = os.getenv("WORKBUDDY_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("WORKBUDDY_MODEL", "deepseek-chat")

RULE = "─" * 78


def _short(obj, limit: int = 80) -> str:
    return clip_chars(json.dumps(obj, ensure_ascii=False, default=str), limit)


class MemoryReActAgent:
    def __init__(self, client, model: str = MODEL, session_id: str = None,
                 task_id: str = None, verbose: bool = True, echo_trace: bool = False,
                 max_steps: int = config.REACT_MAX_STEPS):#echo_trace是否把追踪记录详细打印
        self.client = client
        self.model = model#模型名称
        self.session_id = session_id or f"s-{datetime.now():%Y%m%d-%H%M%S}"#会话ID
        self.requested_task_id = task_id#任务ID
        self.verbose = verbose#是否打印详细日志
        self.max_steps = max_steps#最大步数

        config.ensure_dirs()#自动创建需要的文件夹，用来存记忆，追踪文件
        self.store = Store()#存储管理器
        self.tracer = Tracer(self.store, self.session_id, None, echo=echo_trace)#追踪记录器
        self.embedder = Embedder(client=client, tracer=self.tracer)#嵌入生成器，用于生成嵌入向量，把文本转成embedding
        self.states = TaskStateManager(self.store, self.tracer)#任务状态管理器
        self.ltm = LongTermMemory(self.store, self.embedder, self.tracer)#长期记忆管理器
        self.assembler = ContextAssembler(T.TOOL_DOCS, self.tracer)#上下文组装器
        self.stm = ShortTermMemory(self.store, self.session_id, None, self.tracer)#短期记忆管理器

        self.state = None
        self.resumed = False
        self.last_tool_events = []
        self.last_steps = 0
        self.last_tool_calls = 0
        self.last_package = None

    # ------------------------------------------------------------------ 工具
    def _log(self, *parts):
        if self.verbose:
            print(*parts)

    def _call_llm(self, messages):
        with self.tracer.span("llm.call", model=self.model,
                              tokens=estimate_tokens(messages[0]["content"])
                              + estimate_tokens(messages[1]["content"])) as extra:
            kwargs = dict(model=self.model, messages=messages,
                          temperature=0.0, stop=config.STOP_SEQUENCES)
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                if "stop" in str(exc).lower():
                    kwargs.pop("stop", None)
                    resp = self.client.chat.completions.create(**kwargs)
                else:
                    raise
            if resp is None or not resp.choices:
                # 网关偶发空响应（评测里真实撞到过一次：resp 直接是 None）：
                # 重试一次再判失败，别让一次抖动就把整条任务判死
                time.sleep(1.0)
                resp = self.client.chat.completions.create(**kwargs)
            if resp is None or not resp.choices:
                raise RuntimeError("模型连续两次没有返回任何 choices")
            content = (resp.choices[0].message.content or "").strip()
            extra["reply_chars"] = len(content)
            return content

    # ------------------------------------------------------------------ 任务切换
    def _new_task_id(self) -> str:
        n = len(self.store.state_list(limit=500)) + 1
        return f"t-{datetime.now():%m%d-%H%M%S}-{n:02d}"

    def _ensure_state(self, goal: str):
        if self.state is not None and self.state.status != STATUS_DONE:
            return
        task_id = None
        if self.state is None and self.requested_task_id:
            task_id = self.requested_task_id
        elif self.state is not None:
            task_id = self._new_task_id()
        else:
            task_id = self._new_task_id()

        state, resumed = self.states.load_or_create(task_id, self.session_id, goal)
        self.state = state
        self.resumed = resumed
        self.tracer.bind(task_id=state.task_id)
        self.stm.task_id = state.task_id

        if resumed:
            report = self.states.verify(state)
            self.states.save(state)
            self._log(f"\n[恢复任务 {state.task_id}] 版本 v{state.version}，状态 {state.status}")
            self._log(f"  校验结果：{'一致' if report['ok'] else '不一致 → 相关进度已标记 unknown'}")
            for item in report["mismatches"][:3]:
                self._log(f"    ✗ {item}")
            for item in report["unknown"][:3]:
                self._log(f"    ? {item}")
            for item in report["matches"][:3]:
                self._log(f"    ✔ {item}")
        else:
            self._log(f"\n[新任务 {state.task_id}] 目标：{goal}")
            self._log(f"  验收标准（自动推导）：{'；'.join(state.acceptance)}")
            self.states.save(state)

    # ------------------------------------------------------------------ 单轮
    def _read_state(self):
        """每轮开始前读一次磁盘上的 checkpoint（只看版本号，避免覆盖内存里更新的状态）。"""
        row = self.store.state_load(self.state.task_id)
        disk_version = row["version"] if row else None
        self.tracer.event("state.read", task_id=self.state.task_id,
                          memory_version=self.state.version, disk_version=disk_version)
        if row and disk_version and disk_version > self.state.version:
            from harness.task_state import TaskState
            self.state = TaskState.from_dict(json.loads(row["payload"]))
            self.tracer.event("state.refresh", task_id=self.state.task_id,
                              version=self.state.version, reason="磁盘上的版本更新")
        return self.state

    def ask(self, user_input: str) -> str:
        self._ensure_state(user_input)
        state = self.state

        self.stm.begin_turn()
        self.stm.add("user", user_input, protected=True)
        self.last_tool_events = []
        self.last_steps = 0
        self.last_tool_calls = 0
        last_signature, repeat = None, 0

        for step in range(1, self.max_steps + 1):
            self.last_steps = step
            self.tracer.bind(step=step)
            self.stm.set_step(step)

            # ① 读任务快照（每轮都读一次，防止外部改过 checkpoint 而我不知道）
            state = self._read_state()

            # ② 召回长期记忆（用当前任务做 query，不是用整段历史）
            query = " ".join(dict.fromkeys([state.goal, user_input,
                                            " ".join(t.text for t in state.todos
                                                     if t.status != "done")]))
            memories = self.ltm.recall(query)

            # ③ 按预算装配上下文（每轮重建，不追加）
            pkg = self.assembler.build(state=state, memories=memories, stm=self.stm,
                                       current=user_input, query=query)
            self.last_package = pkg
            self._print_context(step, pkg, memories)

            # ④ 模型决策
            raw = self._call_llm(pkg.messages)
            parsed = parse_react_output(raw)

            if parsed.thought:
                self._log(f"  Thought     : {parsed.thought}")

            # ⑤ 收尾
            if parsed.final is not None:
                self._log("  (模型判断信息足够 → 终止循环)")
                return self._finish(parsed.final, user_input)

            # ⑥ 解析失败 → 当成 Observation 回灌，让它自我纠正
            if parsed.error and parsed.action_input is None:
                self._log(f"  [!] 解析失败：{parsed.error}")
                self.stm.add("note", f"格式错误：{parsed.error}", protected=False,
                             meta={"kind": "parse_error"})
                self.stm.add("observation", json.dumps(
                    {"success": False, "error": f"{parsed.error}。请重新按 ReAct 格式输出。"},
                    ensure_ascii=False), meta={"tool": "parser", "success": False})
                continue

            # ⑦ 执行 Action
            action, arguments = parsed.action or "", parsed.action_input or {}
            signature = f"{action}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
            if signature == last_signature:
                repeat += 1
            else:
                repeat = 0
            last_signature = signature

            if repeat >= 2:
                obs = json.dumps({"success": False,
                                  "error": "相同参数连续调用已阻止，请换策略或直接作答。"},
                                 ensure_ascii=False)
                cost, result = 0.0, {"success": False}
            else:
                obs, cost, result = T.execute_action(action, arguments)
                self.last_tool_calls += 1

            self._log(f"  Action      : {action}")
            self._log(f"  Action Input: {json.dumps(arguments, ensure_ascii=False)}")
            self._log(f"  Observation : {obs}" + (f"  ({cost:.0f} ms)" if cost else ""))

            row_id = self._row_id(action, result)
            ok = bool(result.get("success"))
            self.tracer.event("tool.exec", tool=action, ok=ok, args=arguments,
                              ms=round(cost, 1), row_id=row_id, obs_chars=len(obs))
            if parsed.thought:
                self.stm.add("thought", parsed.thought, meta={"step": step})
            self.stm.add("action", f"{action} {json.dumps(arguments, ensure_ascii=False)}",
                         meta={"tool": action, "args": arguments, "row_id": row_id})
            self.stm.add("observation", obs,
                         meta={"tool": action, "success": ok, "row_id": row_id})

            # ⑧ 更新 checkpoint（动作 + 产物），立刻落盘
            artifact = self._artifact_for(action, ok, row_id)
            summary = self._summary(action, result)
            state.record_action(action, arguments, ok, summary, artifact, row_id)
            if not ok:
                state.add_question(f"{action} 失败：{result.get('error')}")
            self.states.save(state)
            self.last_tool_events.append({"tool": action, "args": arguments, "ok": ok,
                                          "row_id": row_id, "summary": summary})

        # 步数用尽：强制收尾
        self._log(f"\n[达到最大步数 {self.max_steps}] 强制收尾")
        return self._force_final(user_input)

    # ------------------------------------------------------------------ 收尾
    def _finish(self, answer: str, user_input: str, completed: bool = True) -> str:
        state = self.state
        self.stm.add("answer", answer, protected=True)
        if completed:
            if not state.todos:
                todo = state.add_todo(f"完成用户请求：{state.goal[:60]}")
                todo.status = "done"
                todo.evidence = list(state.artifacts[-3:])
            else:
                for todo in state.todos:
                    if todo.status in {"in_progress", "pending"} and state.artifacts:
                        todo.status = "done"
                        todo.evidence = list(dict.fromkeys(todo.evidence + state.artifacts[-2:]))
            state.status = STATUS_DONE
        else:
            state.status = STATUS_RUNNING
            state.add_question("本轮达到最大步数仍未完成，需要下轮继续或人工介入")
        self.states.save(state)

        written = self.ltm.extract_and_store(user_input, answer, self.last_tool_events,
                                             state.task_id, self.session_id)
        if written:
            self._log("\n[长期记忆] 本次新写入：" + "；".join(
                f"#{w['id']}({w.get('superseded') and '取代 #%s' % w['superseded'] or '新增'})"
                for w in written))
        self.tracer.event("react.finish", task_id=state.task_id, steps=self.last_steps,
                          tool_calls=self.last_tool_calls, answer=answer[:120],
                          ltm_written=[w.get("id") for w in written])
        return answer

    def _force_final(self, user_input: str) -> str:
        pkg = self.last_package
        note = ("\n\n（已达到最大循环步数，请不要再调用工具。直接输出 Thought 和 Final Answer，"
                "并说明哪些部分尚未完成。）")
        messages = [dict(m) for m in pkg.messages]
        messages[1] = {"role": "user", "content": messages[1]["content"] + note}
        raw = self._call_llm(messages)
        parsed = parse_react_output(raw)
        answer = parsed.final or raw.strip()
        return self._finish(answer, user_input, completed=False)

    # ------------------------------------------------------------------ 小工具
    @staticmethod
    def _row_id(action, result):
        if not result or not result.get("success"):
            return None
        if action == "add_student":
            return (result.get("data") or {}).get("id")
        if action == "update_student":
            return (result.get("after") or result.get("before") or {}).get("id")
        if action == "delete_student":
            return (result.get("deleted") or {}).get("id")
        return None

    @staticmethod
    def _artifact_for(action, ok, row_id):
        if not ok or not row_id:
            return None
        try:
            if action == "delete_student":
                return f"!{T.TABLE}#id={row_id}"     # 期望"不存在"，恢复校验时反向核对
            if action in {"add_student", "update_student"}:
                return T.make_artifact(row_id)
        except Exception:
            return None
        return None

    @staticmethod
    def _summary(action, result):
        if not result:
            return action
        if result.get("success"):
            return f"{action}: {result.get('message', '成功')}"
        return f"{action} 失败: {result.get('error', '未知错误')}"

    def _print_context(self, step, pkg, memories):
        if not self.verbose:
            return
        b = pkg.budget
        print(f"\n{RULE}")
        print(f"[第 {step} 轮] 装配上下文 → 调模型（{pkg.total_tokens}/{b['limit']} tok）")
        print(f"  预算明细：规则 {b['core']} + 任务状态 {b['state']} + 长期记忆 {b['long_memory']}"
              f" + 短期轨迹 {b['short_memory']} + 当前指令 {b['current']}")
        if memories:
            print("  长期记忆召回：" + "；".join(
                f"#{m.id} {m.kind}({m.sim:.2f})" + ("[可能过时]" if m.stale else "")
                for m in memories))
        else:
            print("  长期记忆召回：无")
        print(f"  短期记忆：{pkg.stm_block.brief()}")
        if pkg.degraded:
            print("  短期记忆已触发降级（去重/剔低相关/摘要压缩）")
        st = self.state
        print(f"  任务状态：{st.task_id} v{st.version} {st.status}"
              f"（动作 {len(st.completed_actions)}，产物 {len(st.artifacts)}）")
        print(RULE)


# ==========================================================================
# 启动 / 交互
# ==========================================================================

def check_llm_config():
    problems = []
    if not API_KEY or "替换" in API_KEY:
        problems.append("API Key 未配置：复制 .env.example 为 .env 并填写 WORKBUDDY_API_KEY")
    if not BASE_URL or "your-workbuddy-host" in BASE_URL:
        problems.append("接口地址未配置：改 BASE_URL（OpenAI 兼容地址，一般以 /v1 结尾）")
    if not MODEL or "your-model-name" in MODEL:
        problems.append("模型名未配置：改 MODEL 为接口支持的模型名")
    return problems


def check_db():
    try:
        fp = T.db_fingerprint()
        print(f"[信息] 学生库正常：{fp['table']} 共 {fp['row_count']} 条，最大 id={fp['max_id']}")
        return True
    except Exception as exc:
        print(f"[警告] 学生库连接失败：{exc}")
        return False


def print_memory_overview(agent):
    s = agent.store
    print(f"[记忆] 短期记忆 {agent.stm.stats()}")
    ltm = agent.ltm.stats()
    print(f"[记忆] 长期记忆 total={ltm['total']} by_kind={ltm['by_kind']} "
          f"superseded={ltm['superseded']} backend={ltm['backend']}")
    tasks = s.state_list(limit=5)
    if tasks:
        print("[记忆] 最近任务：")
        for t in tasks:
            print(f"   {t['task_id']}  {t['status']}  v{t['version']}  {t['goal'][:40]}")


def show_task(agent, task_id):
    row = agent.store.state_load(task_id)
    if not row:
        print(f"没有任务 {task_id}")
        return
    from harness.task_state import TaskState
    state = TaskState.from_dict(json.loads(row["payload"]))
    print(state.render())
    print("\n[checkpoint 文件] " + os.path.join(config.TASK_DIR, f"{task_id}.json"))
    print("[轨迹]")
    for ev in agent.store.trace_query(task_id=task_id, limit=30):
        detail = json.loads(ev["detail"] or "{}")
        print(f"  #{ev['id']:>4} {ev['ts'][11:]} step={ev['step']} {ev['event']}"
              f" {_short(detail, 120)}")


def main():
    parser = argparse.ArgumentParser(description="带三层记忆的 ReAct 学生库 Agent")
    parser.add_argument("--task", help="指定任务 id（存在则恢复，不存在则新建）")
    parser.add_argument("--session", help="指定会话 id（默认按时间生成）")
    parser.add_argument("--trace", action="store_true", help="打印每条 trace 事件")
    parser.add_argument("--show", metavar="TASK_ID", help="打印某个任务的 checkpoint 与 trace")
    parser.add_argument("--stats", action="store_true", help="打印记忆统计后退出")
    args = parser.parse_args()

    print("=" * 78)
    print(" 学生库 Agent · ReAct + 三层记忆（短期 / 结构化状态 / 长期向量）")
    print("=" * 78)

    offline = bool(args.stats or args.show)
    client = None
    if not offline:
        problems = check_llm_config()
        if problems:
            for p in problems:
                print(f"[配置错误] {p}")
            print("\n请先配置，或用环境变量传入后重试。")
            sys.exit(1)

    if not offline:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60, max_retries=2)

    agent = MemoryReActAgent(client=client, model=MODEL, session_id=args.session,
                             task_id=args.task, echo_trace=args.trace)

    print(f"[信息] 模型：{MODEL}    session={agent.session_id}")
    check_db()
    print_memory_overview(agent)

    if args.stats:
        return
    if args.show:
        show_task(agent, args.show)
        return

    print("\n命令：exit 退出 | clear 清空本会话短期记忆并开新任务 | state 看当前 checkpoint"
          " | mem 看记忆统计 | trace 看最近 trace")
    print("-" * 78)

    while True:
        try:
            question = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if not question:
            continue
        low = question.lower()
        if low in {"exit", "quit", "q", "bye"} or question in {"退出", "再见"}:
            print("已退出。")
            break
        if low in {"clear", "reset"} or question in {"清空", "重置"}:
            n = agent.store.stm_clear(agent.session_id)
            agent.stm.load()
            agent.state = None
            agent.requested_task_id = None
            print(f"已清空短期记忆 {n} 条，下次提问会开新任务。")
            continue
        if question in {"state", "状态"}:
            print(agent.state.render() if agent.state else "当前还没有任务")
            continue
        if question in {"mem", "记忆"}:
            print_memory_overview(agent)
            continue
        if question in {"trace", "日志"}:
            tid = agent.state.task_id if agent.state else None
            for ev in agent.store.trace_query(task_id=tid, limit=25):
                detail = json.loads(ev["detail"] or "{}")
                print(f"  #{ev['id']:>4} {ev['ts'][11:]} step={ev['step']} {ev['event']}"
                      f" {_short(detail, 110)}")
            continue

        try:
            started = time.perf_counter()
            answer = agent.ask(question)
            cost = time.perf_counter() - started
        except KeyboardInterrupt:
            print("\n已取消本轮。")
            continue
        except Exception as exc:
            print(f"\n[错误] {type(exc).__name__}: {exc}")
            print("       检查网络 / BASE_URL / MODEL；记忆与 checkpoint 已保存，可重试。")
            continue

        print(f"\n{RULE}\n[最终回答]\n{answer}\n{RULE}")
        print(f"[统计] {agent.last_steps} 轮思考，{agent.last_tool_calls} 次工具调用，"
              f"上下文 {agent.last_package.total_tokens} tok，耗时 {cost:.1f}s")
        print(f"[任务] {agent.state.task_id} {agent.state.status}"
              f" | checkpoint: data/tasks/{agent.state.task_id}.json")


if __name__ == "__main__":
    main()
