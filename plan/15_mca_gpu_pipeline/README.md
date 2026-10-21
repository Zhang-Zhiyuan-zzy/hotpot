# Plan 15：MCA GPU 流水线

状态：已完成。

工作分支：`perf/mca-gpu-pipeline`

基线提交：`88ccec5`

本阶段用一套显式 CPU/GPU 接口替换旧 `MCAPredictor(device=...)` 路径，
把 GPU 推理改造成有界、可流式、可多卡扩展的三段流水线，同时保持现有
MCA 数值语义、位点顺序及 `is_Nuc_site` 判定不变。

后续追加优化把离散窗口 barrier 改为双门控滚动准入，并为显式 GPU
调用增加 256 个预测位点的启动阈值。CLI 仍默认 CPU；显式 GPU 是 GPU-preferred：
整次调用少于阈值时静默使用 CPU，达到阈值才启动 GPU，启动后的错误不回退。
滚动准入以 preprocessing credit 和 resident hard cap 双门控，不增加 worker、
CUDA session、request queue 或 GPU in-flight；resident 上限保持在旧实现的名义
资源包络内，业务输出保持不变。

- [实施方案](mca_gpu_pipeline_implementation_plan.md)
- [实施日志](implementation_log.md)
- [实施报告](implementation_report.md)

本计划不引入 profile。这里的 profile 指“一组有名字的参数预设”，例如
`throughput`、`balanced`、`low_memory`。它会让同一入口因预设名产生隐含配置，
增加学习、复现和排错成本。本阶段直接保留经验证的默认值，并允许用户通过
原子参数覆盖。

