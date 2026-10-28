# Stage 21 — MetalCluster 与多连接位点超原子

状态：2026-10-28 框架草稿；具体实施留白。

本阶段规划注册式 MetalCluster：内部表达图或点云，外部作为具有多个连接位点的
超原子使用。当前只登记类型及与 Molecule、Atom、Bond 的关系；具体结构、匹配
通式及首个注册定义由用户后续给出。

- [实施计划草稿](metal_cluster_implementation_plan.md)
- [Plan20 架构与类型关联](../20_fact_rule_architecture/fact_rule_architecture_implementation_plan.md)
- [化学类型公开接口一致协议](../20_fact_rule_architecture/chemical_type_interface_contract.md)

未来入口为 `Molecule.metal_clusters -> list[MetalCluster]`；本次文档登记没有建立生产类、
注册数据或可调用接口，也不构成 Plan20 的新增实施前置。
