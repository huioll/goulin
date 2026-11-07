# 评测（eval）

两套互相独立的评测，回答两个不同的问题：

| 文件 | 测什么 | 要不要调模型 | 产出 |
|---|---|---|---|
| `eval_run.py` | Agent 完成任务靠不靠谱 | 要（deepseek-chat） | 20 任务成功率 |
| `eval_budget.py` | 记忆机制在长会话下稳不稳定 | 不要（离线） | 预算 / 降级 / 证据保留数据 |

## 运行

```bash
./venv/Scripts/python.exe eval/eval_budget.py          # 离线，几秒
./venv/Scripts/python.exe eval/eval_run.py             # 全量 20 条，约 5-10 分钟
./venv/Scripts/python.exe eval/eval_run.py T07 T18     # 只跑指定任务
```

## 隔离（都不碰真实数据）

- MySQL 用独立表 `student_scores_eval`（`LIKE student_scores` 克隆 + 12 行种子，跑完 `DROP`）
- SQLite 记忆库 / checkpoint / trace 通过 `HARNESS_DB` / `HARNESS_DATA_DIR` 重定向到 `eval/data/`（已 gitignore）
- 种子数据在 `eval_tasks.py` 的 `SEED_ROWS`，所有期望值由它推出

## 任务构成（20 条）

| 类别 | 条数 | 例子 |
|---|---|---|
| 单工具查询 | 6 | 查某学生单科分、班级人数、均分、最高分、总分 |
| 多步 | 4 | 查后加 5 分更新、找最低分再改、两生对比、两班人数 |
| 写后读 | 4 | 新增后回查、更新后确认、批量新增后统计 |
| 删除校验 | 3 | 删除、删除后确认、删不存在的学号 |
| 记忆续用 | 3 | 同会话内"记住规则→按规则算"、"刚才改了谁"、"偏好默认班级新增" |

判定标准：`eval_tasks.py` 每条任务带 `check(answers, db)`，同时核对**最终答案**与**数据库终态**，不是只看模型嘴上说什么。

## 结果

`results/` 下按时间戳保存 JSON（成功明细、每任务步数 / 工具调用 / 峰值 token / 耗时）。
最新一次的结果摘要在主 README 的「评测」一节。
