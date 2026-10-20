# Plan 15 实施日志

## 记录规则

本日志随实施更新，不在完成后倒填。每个阶段记录计划结果、实际工作、验证证据、
意外发现、方案调整及对应提交。

只修改独立 worktree `/tmp/hotpot-mca-gpu-pipeline` 中的
`perf/mca-gpu-pipeline` 分支；主仓库现有未提交内容不属于本阶段。

## 基线

| 项目 | 结果 |
| --- | --- |
| 日期 | 2026-10-20 |
| 工作分支 | `perf/mca-gpu-pipeline` |
| 独立 worktree | `/tmp/hotpot-mca-gpu-pipeline` |
| 起始提交 | `88ccec5` |
| 上一轮 MCA 优化 | `d8138aa` |
| 基线 MCA 测试 | `77 passed in 14.15s` |

## 阶段记录

### 阶段 0 — 固化已批准方案

状态：完成

已完成：

- 从主仓库当前提交建立独立分支与 worktree；
- 明确不引入 profile；直接保留每卡 9/7 worker 默认值；
- 明确 GPU 选择、CPU 数量推导、显存 batch 档位及 OOM 行为；
- 明确删除旧接口与旧 GPU 后端，不设置兼容开关；
- 固化 all-atom、logical atom-row batch、padding、顺序和 `is_Nuc_site` 边界；
- 创建实施方案、实施日志和最终报告入口。
- 执行全部现有 MCA 与 calculator 集成测试，确认改动前基线为 77 项通过。

验证：

- `git status --short --branch` 显示当前分支为 `perf/mca-gpu-pipeline`；
- `git log -1` 显示起点为 `88ccec5`。
- `PYTHONPATH=/tmp/hotpot-onnx-inspect:. python -m pytest -q tests/mca tests/test_cheminfo/test_mca_calculator.py`
- 结果：`77 passed in 14.15s`。

意外发现：

- 新 worktree 不包含 Git 忽略的 CPython native extension，首次收集因此失败；
- 从同一源提交的主 worktree 复制三个 CPython 3.11 构建产物后恢复测试。这些
  `.so` 仅用于本地验证，不纳入提交，也不改变产品代码或测试范围。

方案调整：无。

提交：待写入 Plan 15 文档检查点提交号。

### 阶段 1 — 资源、显存与 batching 原语

状态：待开始

