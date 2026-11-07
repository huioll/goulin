# -*- coding: utf-8 -*-
"""Harness 离线回归测试（标准库 unittest，零额外依赖）。

设计原则：不连 MySQL、不调大模型、不写真实 data/ 目录——
全部用临时 SQLite + 假数据驱动，`python -m unittest discover -s tests -v` 即可跑。

六个用例分别对应设计里的六个断言点：
    token 估算        —— 预算控制的地基，宁可高估也不撑爆窗口
    降级阶段单调      —— 四级降级必须按 raw→dedupe→drop_low_rel→compress→shrink 走
    保护记录不丢      —— 降级只压非保护内容，关键证据必须留得住
    产物引用校验      —— checkpoint 的哈希能识别"外部改过"
    长期记忆取代      —— 相似结论覆盖旧结论，而不是无限堆积
    上下文预算装配    —— 每轮 messages 只有 system + user 两条，且总量不超预算
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness import config
from harness.context import ContextAssembler
from harness.embedding import Embedder
from harness.long_term import LongTermMemory
from harness.short_term import ShortTermMemory
from harness.store import Store
from harness.task_state import TaskState, TaskStateManager
from harness.tokens import estimate_tokens

STAGE_ORDER = ["raw", "dedupe", "drop_low_rel", "compress", "shrink"]


class HarnessTestCase(unittest.TestCase):
    """每个用例一份独立的临时库，避免相互污染，也绝不碰 data/harness.db。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="harness-test-")
        self.store = Store(os.path.join(self.tmp, "test.db"))
        self._orig_task_dir = config.TASK_DIR
        config.TASK_DIR = os.path.join(self.tmp, "tasks")
        os.makedirs(config.TASK_DIR, exist_ok=True)

    def tearDown(self):
        config.TASK_DIR = self._orig_task_dir
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestTokenEstimate(HarnessTestCase):

    def test_cjk_counts_more_than_latin(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens(None), 0)

        zh = "学生的语文成绩记录"          # 9 个中文字
        en = "student score record"       # 20 个拉丁字符

        # 中文按 1 字 ≈ 1 token，估算值不会低于字数
        self.assertGreaterEqual(estimate_tokens(zh), len(zh))
        # 英文按约 3.5 字符 ≈ 1 token，同样长度下明显更省
        self.assertLess(estimate_tokens(en), len(en))
        self.assertGreater(estimate_tokens(zh), estimate_tokens(en))


class TestShortTermDegrade(HarnessTestCase):

    def _fill(self, stm, n, protected_first=False):
        if protected_first:
            stm.add("observation", "关键证据：记录 id=1 的语文成绩为 91.5",
                    protected=True, meta={"tool": "get_student", "row_id": 1})
        for i in range(n):
            stm.add("observation", f"第 {i} 条观测：" + "填充内容" * 25,
                    meta={"tool": "search_students"})

    def test_stages_monotonic_and_compression_real(self):
        stm = ShortTermMemory(self.store, session_id="s-degrade")
        self._fill(stm, 30)

        block = stm.build_block(query="查一下学生成绩", budget_tokens=200)

        stages = [s["stage"] for s in block.stages]
        self.assertEqual(stages[0], "raw", "降级链首项必须是未处理的基线")

        # 阶段必须按定义顺序推进，不允许跳级或回退
        idx = [STAGE_ORDER.index(s) for s in stages]
        self.assertEqual(idx, sorted(idx), f"降级阶段顺序异常：{stages}")

        # 确实压掉了大半内容
        self.assertLess(block.used_tokens, block.before_tokens * 0.5,
                        f"压缩力度不足：{block.before_tokens} → {block.used_tokens}")

        # 预算是软目标：最近窗口 STM_KEEP_RECENT 条永远不裁，
        # 所以最终条数收敛到「摘要 + 最近窗口 + 少量保护记录」，而不是压到 0
        self.assertLessEqual(block.rendered, config.STM_KEEP_RECENT + 3,
                             f"降级后仍保留 {block.rendered} 条，最近窗口策略未生效")

    def test_protected_record_survives_degrade(self):
        stm = ShortTermMemory(self.store, session_id="s-protect")
        self._fill(stm, 40, protected_first=True)
        keep_id = stm.records[0].id

        block = stm.build_block(query="查学生成绩", budget_tokens=120)

        # 预算压得很紧，但被保护的证据必须仍然独立存在、可追溯
        evidence = " ".join(block.kept_evidence)
        self.assertIn(str(keep_id), evidence,
                      f"保护记录 #{keep_id} 在降级中丢失，保留证据为 {block.kept_evidence}")

        # 内容也不能被裁成半截——保护记录应原样保留在最终上下文里
        self.assertIn("91.5", block.text, "保护记录的原始内容被裁短了")


class TestCheckpointArtifact(HarnessTestCase):
    """直接用假的 fetch_row 驱动 _check_ref，完全不碰数据库。"""

    def _ref(self, ref):
        import tools
        fetch = lambda rid: {"id": rid, "student_name": "小明"} if rid == 1 else None
        return TaskStateManager._check_ref(
            ref, fetch, lambda row: "abcd1234", tools.parse_artifact, tools.TABLE
        )

    def test_ref_verify_matrix(self):
        # 行还在，且哈希一致 → ok
        self.assertEqual(self._ref("student_scores#id=1@abcd1234")[0], "ok")
        # 行还在，但内容被外部改过 → mismatch
        self.assertEqual(self._ref("student_scores#id=1@deadbeef")[0], "mismatch")
        # 快照里有这行，现场没了 → mismatch
        self.assertEqual(self._ref("student_scores#id=9@abcd1234")[0], "mismatch")
        # 删除类产物：期望不存在，现场确实没有 → ok
        self.assertEqual(self._ref("!student_scores#id=9")[0], "ok")
        # 删除类产物：期望不存在，但现场还在 → mismatch
        self.assertEqual(self._ref("!student_scores#id=1")[0], "mismatch")


class TestLongTermSupersede(HarnessTestCase):

    def test_conflicting_fact_supersedes_old(self):
        ltm = LongTermMemory(self.store, Embedder())

        ltm.remember("学生的语文满分是150分", kind="fact")
        ltm.remember("学生的语文满分是120分", kind="fact")

        rows = self.store.ltm_all(include_dead=True)
        self.assertEqual(len(rows), 2, "两条结论都应入库")

        alive = [r for r in rows if r["superseded_by"] is None]
        dead = [r for r in rows if r["superseded_by"] is not None]

        # 同一句话只改了数值，旧的必须被新结论取代，而不是并存
        self.assertEqual(len(alive), 1, f"仍有 {len(alive)} 条结论并存，取代未生效")
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0]["text"], "学生的语文满分是150分")
        self.assertEqual(alive[0]["text"], "学生的语文满分是120分")

        # 被取代的条目不应再出现在召回结果里
        hits = ltm.recall("语文满分多少", top_k=5)
        self.assertNotIn(dead[0]["text"], [h.text for h in hits])


class TestContextAssembly(HarnessTestCase):

    def test_messages_rebuilt_and_budget_respected(self):
        asm = ContextAssembler(tool_docs="可用工具：get_student / search_students / add_student")
        state = TaskState(task_id="t-ctx", session_id="s-ctx", goal="查询学生成绩")

        stm = ShortTermMemory(self.store, session_id="s-ctx", task_id="t-ctx")
        for i in range(20):
            stm.add("observation", f"记录{i}：" + "内容" * 40, meta={"tool": "search_students"})

        pkg = asm.build(state=state, memories=[], stm=stm,
                        current="帮我查一下高三一班的语文成绩", query="查高三一班语文成绩")

        # 核心断言：每轮重建，messages 永远只有 system + user 两条，不带历史
        self.assertEqual(len(pkg.messages), 2,
                         f"messages 应只有 system+user，实际 {len(pkg.messages)} 条")
        self.assertEqual(pkg.messages[0]["role"], "system")
        self.assertEqual(pkg.messages[1]["role"], "user")

        # 总量不得突破总预算（留 5% 容差，避免估算粒度导致的边界抖动）
        self.assertLessEqual(pkg.total_tokens, int(config.TOKEN_BUDGET_TOTAL * 1.05),
                             f"装配后 {pkg.total_tokens} 超出预算 {config.TOKEN_BUDGET_TOTAL}")

        # 五块配额都要有记录，便于事后核对是哪一块挤爆的
        self.assertEqual(len(pkg.blocks), 5, f"应有五块配额，实际 {list(pkg.blocks)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
