# 707个执行单元的业务映射

这是现有Task506清单的逐项展开，不是新增验证计划，也不是707项业务已经通过的声明。原始清单包含707个唯一command ID；700条为70种探针配方×5个Python版本×源码/安装包两种形态，另有5条安装包core冒烟和2条CP310递归反射补充。一个单元可能内部包含多个断言或分子，因此不能把707当成独立化学能力数。

优先级沿用用户2026-11-03定义：1分子表示与转化；2化学规则/语义；3 CBond→FF→GFN-FF→xTB及几何门控；4 MCA；5 I/O；6 RDKit/Open Babel通讯（业务接口）；7其他第三方交互。斜杠分隔的多个数字表示关联范围，不相加统计。

本表标识的是计划观测目标；没有读取增长中的业务payload来宣布通过。3和4没有完整实际工作流执行单元。固定六分子夹具为`CCO`、`[NH4+]`、`c1ccccc1`、`F[C@](Cl)(Br)I`、`F/C=C/F`、`[13CH3][O-]`，不能代替金属配位、多种价态、真实优化或模型推理覆盖。转换/I/O组使用arange(N*3)/8构造的固定人工坐标，用于运输检查，不是物理合理几何样本。

## 按家族汇总

| 家族 | 全部单元 | Python3.11单元 | 配方/展开 | 直接业务范围与限制 |
|---|---:|---:|---|---|
| cold | 170 | 34 | 17×5×2 | 15个模块冷导入+2种OB/RD顺序；主要检查依赖和可用性 |
| missing-use | 320 | 64 | 32×5×2 | 16种动作/依赖/夹具组合×两个屏蔽时点；检查局部失败与输入保护 |
| cold-unpickle | 70 | 14 | 7×5×2 | 6个旧分子+1个晶胞pickle；事实恢复和导入顺序 |
| reflection | 10 | 2 | 1×5×2 | 签名/注解/默认值；无完整化学计算 |
| preservation | 60 | 12 | 6×5×2 | surface/conversions/io/crystal/thermo/drawing六组；其中两组属7 |
| io-readback | 60 | 12 | 6×5×2 | SMI/MOL2/SDF×单/多记录；Path/字符串读回 |
| drawing-resource | 10 | 2 | 1×5×2 | 14个绘图/资源子案例，属7 |
| installed-core-smoke | 5 | 1 | 1×5，仅安装包 | 基础图/状态选择及资源冒烟；无FF、GFN-FF、xTB或MCA推理 |
| recursive-reflection-supplement | 2 | 0 | 1×1×2，仅CP310 | 注解类型递归补充；不执行化学业务 |
| 合计 | 707 | 141 | 其余Python版本566条 | 运行组合数，不是化学语义覆盖率 |

按本表的主业务分类，thermo/绘图/字体相关单元共270条：冷导入40、缺包200、preservation20、drawing-resource10。该数量不含core冒烟内附带的字体/文档断言，也不表示这些检查没有价值；它说明当前资源分配与必须优先的1–4项不匹配。

## 72种配方/补充的业务对应

原70种配方保持各自10次展开；core冒烟展开5次；递归反射补充展开2次。下表完整列出72种，再列出707个实际ID。

| 配方/补充ID（去Python与形态前缀） | 次数 | 用户业务项 | 实际动作/对象 | 业务意义 | 证据层级/限制 |
|---|---:|---|---|---|---|
| cold-import-00 | 10 | 1/2/5/6（间接） | hotpot | 化学载体、转换、I/O或晶胞模块可导入；不执行对应完整业务 | 导入/依赖边界 |
| cold-import-01 | 10 | 1/2/5/6（间接） | hotpot.cheminfo.structure | 化学载体、转换、I/O或晶胞模块可导入；不执行对应完整业务 | 导入/依赖边界 |
| cold-import-02 | 10 | 1/2/5/6（间接） | hotpot.cheminfo.structure.crystal | 化学载体、转换、I/O或晶胞模块可导入；不执行对应完整业务 | 导入/依赖边界 |
| cold-import-03 | 10 | 1/2/5/6（间接） | hotpot.cheminfo.operations.conversion | 化学载体、转换、I/O或晶胞模块可导入；不执行对应完整业务 | 导入/依赖边界 |
| cold-import-04 | 10 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.conversion | 化学载体、转换、I/O或晶胞模块可导入；不执行对应完整业务 | 导入/依赖边界 |
| cold-import-05 | 10 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.conversion | 化学载体、转换、I/O或晶胞模块可导入；不执行对应完整业务 | 导入/依赖边界 |
| cold-import-06 | 10 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.crystal | 化学载体、转换、I/O或晶胞模块可导入；不执行对应完整业务 | 导入/依赖边界 |
| cold-import-07 | 10 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.conversion | 化学载体、转换、I/O或晶胞模块可导入；不执行对应完整业务 | 导入/依赖边界 |
| cold-import-08 | 10 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.stereo_transport | 化学载体、转换、I/O或晶胞模块可导入；不执行对应完整业务 | 导入/依赖边界 |
| cold-import-09 | 10 | 1/2/5/6（间接） | hotpot.cheminfo.operations.io | 化学载体、转换、I/O或晶胞模块可导入；不执行对应完整业务 | 导入/依赖边界 |
| cold-import-10 | 10 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.io.formats.openbabel | 化学载体、转换、I/O或晶胞模块可导入；不执行对应完整业务 | 导入/依赖边界 |
| cold-import-11 | 10 | 7 | hotpot.cheminfo.operations.thermo | 第三方热物性/绘图模块可导入；不验证其科学结果 | 导入/依赖边界 |
| cold-import-12 | 10 | 7 | hotpot.cheminfo.adapters.thermo.chemical | 第三方热物性/绘图模块可导入；不验证其科学结果 | 导入/依赖边界 |
| cold-import-13 | 10 | 7 | hotpot.cheminfo.operations.visualization | 第三方热物性/绘图模块可导入；不验证其科学结果 | 导入/依赖边界 |
| cold-import-14 | 10 | 7 | hotpot.cheminfo.adapters.visualization.draw | 第三方热物性/绘图模块可导入；不验证其科学结果 | 导入/依赖边界 |
| adapter-order-ob-then-rd | 10 | 1/6 | hotpot；顺序 ob-then-rd | 甲烷的Open Babel/RDKit转换调用顺序与对象可用性 | 小样本转换 |
| adapter-order-rd-then-ob | 10 | 1/6 | hotpot；顺序 rd-then-ob | 甲烷的Open Babel/RDKit转换调用顺序与对象可用性 | 小样本转换 |
| missing-ob-conversion-openbabel-before-hotpot | 10 | 1/6 | ob-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 分子外部对象转换缺依赖时局部失败、原分子保护 | 故障注入/数据完整性 |
| missing-ob-conversion-openbabel-before-use | 10 | 1/6 | ob-conversion；屏蔽 openbabel；before-use；基础夹具 | 分子外部对象转换缺依赖时局部失败、原分子保护 | 故障注入/数据完整性 |
| missing-pybel-conversion-openbabel-before-hotpot | 10 | 1/6 | pybel-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 分子外部对象转换缺依赖时局部失败、原分子保护 | 故障注入/数据完整性 |
| missing-pybel-conversion-openbabel-before-use | 10 | 1/6 | pybel-conversion；屏蔽 openbabel；before-use；基础夹具 | 分子外部对象转换缺依赖时局部失败、原分子保护 | 故障注入/数据完整性 |
| missing-rd-conversion-rdkit-before-hotpot | 10 | 1/6 | rd-conversion；屏蔽 rdkit；before-hotpot；基础夹具 | 分子外部对象转换缺依赖时局部失败、原分子保护 | 故障注入/数据完整性 |
| missing-rd-conversion-rdkit-before-use | 10 | 1/6 | rd-conversion；屏蔽 rdkit；before-use；基础夹具 | 分子外部对象转换缺依赖时局部失败、原分子保护 | 故障注入/数据完整性 |
| missing-io-read-openbabel-before-hotpot | 10 | 5/6 | io-read；屏蔽 openbabel；before-hotpot；基础夹具 | 文件读写缺Open Babel时局部失败和原分子保护 | 故障注入/数据完整性 |
| missing-io-read-openbabel-before-use | 10 | 5/6 | io-read；屏蔽 openbabel；before-use；基础夹具 | 文件读写缺Open Babel时局部失败和原分子保护 | 故障注入/数据完整性 |
| missing-io-write-openbabel-before-hotpot | 10 | 5/6 | io-write；屏蔽 openbabel；before-hotpot；基础夹具 | 文件读写缺Open Babel时局部失败和原分子保护 | 故障注入/数据完整性 |
| missing-io-write-openbabel-before-use | 10 | 5/6 | io-write；屏蔽 openbabel；before-use；基础夹具 | 文件读写缺Open Babel时局部失败和原分子保护 | 故障注入/数据完整性 |
| missing-crystal-cell-openbabel-before-hotpot | 10 | 1/6 | crystal-cell；屏蔽 openbabel；before-hotpot；基础夹具 | 晶胞外部投影缺依赖时局部失败和六参数事实保护 | 故障注入/数据完整性 |
| missing-crystal-cell-openbabel-before-use | 10 | 1/6 | crystal-cell；屏蔽 openbabel；before-use；基础夹具 | 晶胞外部投影缺依赖时局部失败和六参数事实保护 | 故障注入/数据完整性 |
| missing-thermo-thermo-before-hotpot | 10 | 7 | thermo；屏蔽 thermo；before-hotpot；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-thermo-thermo-before-use | 10 | 7 | thermo；屏蔽 thermo；before-use；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-thermo-pandas-before-hotpot | 10 | 7 | thermo；屏蔽 pandas；before-hotpot；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-thermo-pandas-before-use | 10 | 7 | thermo；屏蔽 pandas；before-use；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-svg-cairosvg-before-hotpot | 10 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-svg-cairosvg-before-use | 10 | 7 | draw-svg；屏蔽 cairosvg；before-use；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-svg-matplotlib-before-hotpot | 10 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-svg-matplotlib-before-use | 10 | 7 | draw-svg；屏蔽 matplotlib；before-use；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-png-cairosvg-before-hotpot | 10 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-png-cairosvg-before-use | 10 | 7 | draw-png；屏蔽 cairosvg；before-use；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-webp-PIL-before-hotpot | 10 | 7 | draw-webp；屏蔽 PIL；before-hotpot；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-webp-PIL-before-use | 10 | 7 | draw-webp；屏蔽 PIL；before-use；基础夹具 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| prior-cold-unpickle-0 | 10 | 1/5 | molecule-0.pickle | 旧Molecule/晶胞持久化数据可恢复、事实可读；不要求pickle字节相同 | 持久化/事实恢复 |
| prior-cold-unpickle-1 | 10 | 1/5 | molecule-1.pickle | 旧Molecule/晶胞持久化数据可恢复、事实可读；不要求pickle字节相同 | 持久化/事实恢复 |
| prior-cold-unpickle-2 | 10 | 1/5 | molecule-2.pickle | 旧Molecule/晶胞持久化数据可恢复、事实可读；不要求pickle字节相同 | 持久化/事实恢复 |
| prior-cold-unpickle-3 | 10 | 1/5 | molecule-3.pickle | 旧Molecule/晶胞持久化数据可恢复、事实可读；不要求pickle字节相同 | 持久化/事实恢复 |
| prior-cold-unpickle-4 | 10 | 1/5 | molecule-4.pickle | 旧Molecule/晶胞持久化数据可恢复、事实可读；不要求pickle字节相同 | 持久化/事实恢复 |
| prior-cold-unpickle-5 | 10 | 1/5 | molecule-5.pickle | 旧Molecule/晶胞持久化数据可恢复、事实可读；不要求pickle字节相同 | 持久化/事实恢复 |
| p04-prior-cell-cold-unpickle | 10 | 1/5 | c61-cell-molecule.pickle | 旧Molecule/晶胞持久化数据可恢复、事实可读；不要求pickle字节相同 | 持久化/事实恢复 |
| complete-reflection | 10 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 公开签名、注解、默认值和类型身份；FF仅常量身份 | 接口兼容；非计算流程 |
| preserve-surface | 10 | 1/6（接口） | surface | 8个载体方法与6个门面身份；FF仅默认常量 | 接口兼容；非计算流程 |
| preserve-conversions | 10 | 1/2/6 | conversions | 6个固定分子的原子、电荷、同位素、芳香与立体运输；外部对象新鲜性/隔离；pickle往返 | 结构运输小样本；非全规则验证 |
| preserve-io | 10 | 1/5/6 | io | CCO在SMI/MOL2/SDF的单/多帧写读、帧数与坐标归属 | I/O小样本；非全格式验证 |
| preserve-crystal | 10 | 1/6 | crystal | 斜晶胞六参数、矩阵投影、外部对象隔离与pickle往返 | 晶胞运输小样本 |
| preserve-thermo | 10 | 7 | thermo | fake Chemical的181属性转发与5条回退路线；无真实热物性计算 | 替身接口合约 |
| preserve-drawing | 10 | 7 | drawing | 字符串/Hotpot/RDKit输入、grid、10格式、5字体 | 第三方输出/资源 |
| io-path-string-smi-single | 10 | 1/5/6 | smi-single.smi；Path/字符串入口 | 3种格式×单/多记录，Path与字符串入口的记录基数及事实 | I/O读回小样本 |
| io-path-string-smi-multi | 10 | 1/5/6 | smi-multi.smi；Path/字符串入口 | 3种格式×单/多记录，Path与字符串入口的记录基数及事实 | I/O读回小样本 |
| io-path-string-mol2-single | 10 | 1/5/6 | mol2-single.mol2；Path/字符串入口 | 3种格式×单/多记录，Path与字符串入口的记录基数及事实 | I/O读回小样本 |
| io-path-string-mol2-multi | 10 | 1/5/6 | mol2-multi.mol2；Path/字符串入口 | 3种格式×单/多记录，Path与字符串入口的记录基数及事实 | I/O读回小样本 |
| io-path-string-sdf-single | 10 | 1/5/6 | sdf-single.sdf；Path/字符串入口 | 3种格式×单/多记录，Path与字符串入口的记录基数及事实 | I/O读回小样本 |
| io-path-string-sdf-multi | 10 | 1/5/6 | sdf-multi.sdf；Path/字符串入口 | 3种格式×单/多记录，Path与字符串入口的记录基数及事实 | I/O读回小样本 |
| drawing-independent-cases | 10 | 7 | 字体、grid、single-string、版本拒绝顺序、10种绘图格式 | 14个独立绘图/字体/版本边界子案例 | 第三方输出/资源 |
| missing-draw-svg-cairosvg-before-hotpot-bonded-drawing-supplement1 | 10 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；有键分子补充 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-svg-cairosvg-before-use-bonded-drawing-supplement1 | 10 | 7 | draw-svg；屏蔽 cairosvg；before-use；有键分子补充 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-svg-matplotlib-before-hotpot-bonded-drawing-supplement1 | 10 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；有键分子补充 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-svg-matplotlib-before-use-bonded-drawing-supplement1 | 10 | 7 | draw-svg；屏蔽 matplotlib；before-use；有键分子补充 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-png-cairosvg-before-hotpot-bonded-drawing-supplement1 | 10 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；有键分子补充 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-png-cairosvg-before-use-bonded-drawing-supplement1 | 10 | 7 | draw-png；屏蔽 cairosvg；before-use；有键分子补充 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-webp-PIL-before-hotpot-bonded-drawing-supplement1 | 10 | 7 | draw-webp；屏蔽 PIL；before-hotpot；有键分子补充 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| missing-draw-webp-PIL-before-use-bonded-drawing-supplement1 | 10 | 7 | draw-webp；屏蔽 PIL；before-use；有键分子补充 | 可选依赖缺失的报错边界、隔离及原事实未被改写 | 故障注入/第三方边界 |
| core-smoke | 5 | 1/2；其余仅资源 | 图环、子结构搜索、甲烷电荷/自旋状态选择、包资源与CLI文档 | 环/子结构/简单电荷自旋的安装包冒烟；MCA/xTB只读CLI文档 | 基础语义冒烟；无FF/xTB/MCA推理 |
| recursive-reflection-supplement | 2 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 公开签名、注解、默认值和类型身份；FF仅常量身份 | 接口兼容；非计算流程 |

## 全部707个执行单元

| 单元ID | Python | 形态 | 用户业务项 | 实际动作/对象 | 证据层级/限制 |
|---|---|---|---|---|---|
| cp310-source-cold-import-00 | 3.10 | 源码 | 1/2/5/6（间接） | hotpot | 导入/依赖边界 |
| cp310-source-cold-import-01 | 3.10 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.structure | 导入/依赖边界 |
| cp310-source-cold-import-02 | 3.10 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.structure.crystal | 导入/依赖边界 |
| cp310-source-cold-import-03 | 3.10 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.operations.conversion | 导入/依赖边界 |
| cp310-source-cold-import-04 | 3.10 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.conversion | 导入/依赖边界 |
| cp310-source-cold-import-05 | 3.10 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.conversion | 导入/依赖边界 |
| cp310-source-cold-import-06 | 3.10 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.crystal | 导入/依赖边界 |
| cp310-source-cold-import-07 | 3.10 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.conversion | 导入/依赖边界 |
| cp310-source-cold-import-08 | 3.10 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.stereo_transport | 导入/依赖边界 |
| cp310-source-cold-import-09 | 3.10 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.operations.io | 导入/依赖边界 |
| cp310-source-cold-import-10 | 3.10 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.io.formats.openbabel | 导入/依赖边界 |
| cp310-source-cold-import-11 | 3.10 | 源码 | 7 | hotpot.cheminfo.operations.thermo | 导入/依赖边界 |
| cp310-source-cold-import-12 | 3.10 | 源码 | 7 | hotpot.cheminfo.adapters.thermo.chemical | 导入/依赖边界 |
| cp310-source-cold-import-13 | 3.10 | 源码 | 7 | hotpot.cheminfo.operations.visualization | 导入/依赖边界 |
| cp310-source-cold-import-14 | 3.10 | 源码 | 7 | hotpot.cheminfo.adapters.visualization.draw | 导入/依赖边界 |
| cp310-source-adapter-order-ob-then-rd | 3.10 | 源码 | 1/6 | hotpot；顺序 ob-then-rd | 小样本转换 |
| cp310-source-adapter-order-rd-then-ob | 3.10 | 源码 | 1/6 | hotpot；顺序 rd-then-ob | 小样本转换 |
| cp310-source-missing-ob-conversion-openbabel-before-hotpot | 3.10 | 源码 | 1/6 | ob-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-ob-conversion-openbabel-before-use | 3.10 | 源码 | 1/6 | ob-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-pybel-conversion-openbabel-before-hotpot | 3.10 | 源码 | 1/6 | pybel-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-pybel-conversion-openbabel-before-use | 3.10 | 源码 | 1/6 | pybel-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-rd-conversion-rdkit-before-hotpot | 3.10 | 源码 | 1/6 | rd-conversion；屏蔽 rdkit；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-rd-conversion-rdkit-before-use | 3.10 | 源码 | 1/6 | rd-conversion；屏蔽 rdkit；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-io-read-openbabel-before-hotpot | 3.10 | 源码 | 5/6 | io-read；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-io-read-openbabel-before-use | 3.10 | 源码 | 5/6 | io-read；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-io-write-openbabel-before-hotpot | 3.10 | 源码 | 5/6 | io-write；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-io-write-openbabel-before-use | 3.10 | 源码 | 5/6 | io-write；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-crystal-cell-openbabel-before-hotpot | 3.10 | 源码 | 1/6 | crystal-cell；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-crystal-cell-openbabel-before-use | 3.10 | 源码 | 1/6 | crystal-cell；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-source-missing-thermo-thermo-before-hotpot | 3.10 | 源码 | 7 | thermo；屏蔽 thermo；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-source-missing-thermo-thermo-before-use | 3.10 | 源码 | 7 | thermo；屏蔽 thermo；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-source-missing-thermo-pandas-before-hotpot | 3.10 | 源码 | 7 | thermo；屏蔽 pandas；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-source-missing-thermo-pandas-before-use | 3.10 | 源码 | 7 | thermo；屏蔽 pandas；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-source-missing-draw-svg-cairosvg-before-hotpot | 3.10 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-source-missing-draw-svg-cairosvg-before-use | 3.10 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-source-missing-draw-svg-matplotlib-before-hotpot | 3.10 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-source-missing-draw-svg-matplotlib-before-use | 3.10 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-source-missing-draw-png-cairosvg-before-hotpot | 3.10 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-source-missing-draw-png-cairosvg-before-use | 3.10 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-source-missing-draw-webp-PIL-before-hotpot | 3.10 | 源码 | 7 | draw-webp；屏蔽 PIL；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-source-missing-draw-webp-PIL-before-use | 3.10 | 源码 | 7 | draw-webp；屏蔽 PIL；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-source-prior-cold-unpickle-0 | 3.10 | 源码 | 1/5 | molecule-0.pickle | 持久化/事实恢复 |
| cp310-source-prior-cold-unpickle-1 | 3.10 | 源码 | 1/5 | molecule-1.pickle | 持久化/事实恢复 |
| cp310-source-prior-cold-unpickle-2 | 3.10 | 源码 | 1/5 | molecule-2.pickle | 持久化/事实恢复 |
| cp310-source-prior-cold-unpickle-3 | 3.10 | 源码 | 1/5 | molecule-3.pickle | 持久化/事实恢复 |
| cp310-source-prior-cold-unpickle-4 | 3.10 | 源码 | 1/5 | molecule-4.pickle | 持久化/事实恢复 |
| cp310-source-prior-cold-unpickle-5 | 3.10 | 源码 | 1/5 | molecule-5.pickle | 持久化/事实恢复 |
| cp310-source-p04-prior-cell-cold-unpickle | 3.10 | 源码 | 1/5 | c61-cell-molecule.pickle | 持久化/事实恢复 |
| cp310-source-complete-reflection | 3.10 | 源码 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |
| cp310-source-preserve-surface | 3.10 | 源码 | 1/6（接口） | surface | 接口兼容；非计算流程 |
| cp310-source-preserve-conversions | 3.10 | 源码 | 1/2/6 | conversions | 结构运输小样本；非全规则验证 |
| cp310-source-preserve-io | 3.10 | 源码 | 1/5/6 | io | I/O小样本；非全格式验证 |
| cp310-source-preserve-crystal | 3.10 | 源码 | 1/6 | crystal | 晶胞运输小样本 |
| cp310-source-preserve-thermo | 3.10 | 源码 | 7 | thermo | 替身接口合约 |
| cp310-source-preserve-drawing | 3.10 | 源码 | 7 | drawing | 第三方输出/资源 |
| cp310-source-io-path-string-smi-single | 3.10 | 源码 | 1/5/6 | smi-single.smi；Path/字符串入口 | I/O读回小样本 |
| cp310-source-io-path-string-smi-multi | 3.10 | 源码 | 1/5/6 | smi-multi.smi；Path/字符串入口 | I/O读回小样本 |
| cp310-source-io-path-string-mol2-single | 3.10 | 源码 | 1/5/6 | mol2-single.mol2；Path/字符串入口 | I/O读回小样本 |
| cp310-source-io-path-string-mol2-multi | 3.10 | 源码 | 1/5/6 | mol2-multi.mol2；Path/字符串入口 | I/O读回小样本 |
| cp310-source-io-path-string-sdf-single | 3.10 | 源码 | 1/5/6 | sdf-single.sdf；Path/字符串入口 | I/O读回小样本 |
| cp310-source-io-path-string-sdf-multi | 3.10 | 源码 | 1/5/6 | sdf-multi.sdf；Path/字符串入口 | I/O读回小样本 |
| cp310-source-drawing-independent-cases | 3.10 | 源码 | 7 | 字体、grid、single-string、版本拒绝顺序、10种绘图格式 | 第三方输出/资源 |
| cp310-installed-cold-import-00 | 3.10 | 安装包 | 1/2/5/6（间接） | hotpot | 导入/依赖边界 |
| cp310-installed-cold-import-01 | 3.10 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.structure | 导入/依赖边界 |
| cp310-installed-cold-import-02 | 3.10 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.structure.crystal | 导入/依赖边界 |
| cp310-installed-cold-import-03 | 3.10 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.operations.conversion | 导入/依赖边界 |
| cp310-installed-cold-import-04 | 3.10 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.conversion | 导入/依赖边界 |
| cp310-installed-cold-import-05 | 3.10 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.conversion | 导入/依赖边界 |
| cp310-installed-cold-import-06 | 3.10 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.crystal | 导入/依赖边界 |
| cp310-installed-cold-import-07 | 3.10 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.conversion | 导入/依赖边界 |
| cp310-installed-cold-import-08 | 3.10 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.stereo_transport | 导入/依赖边界 |
| cp310-installed-cold-import-09 | 3.10 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.operations.io | 导入/依赖边界 |
| cp310-installed-cold-import-10 | 3.10 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.io.formats.openbabel | 导入/依赖边界 |
| cp310-installed-cold-import-11 | 3.10 | 安装包 | 7 | hotpot.cheminfo.operations.thermo | 导入/依赖边界 |
| cp310-installed-cold-import-12 | 3.10 | 安装包 | 7 | hotpot.cheminfo.adapters.thermo.chemical | 导入/依赖边界 |
| cp310-installed-cold-import-13 | 3.10 | 安装包 | 7 | hotpot.cheminfo.operations.visualization | 导入/依赖边界 |
| cp310-installed-cold-import-14 | 3.10 | 安装包 | 7 | hotpot.cheminfo.adapters.visualization.draw | 导入/依赖边界 |
| cp310-installed-adapter-order-ob-then-rd | 3.10 | 安装包 | 1/6 | hotpot；顺序 ob-then-rd | 小样本转换 |
| cp310-installed-adapter-order-rd-then-ob | 3.10 | 安装包 | 1/6 | hotpot；顺序 rd-then-ob | 小样本转换 |
| cp310-installed-missing-ob-conversion-openbabel-before-hotpot | 3.10 | 安装包 | 1/6 | ob-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-ob-conversion-openbabel-before-use | 3.10 | 安装包 | 1/6 | ob-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-pybel-conversion-openbabel-before-hotpot | 3.10 | 安装包 | 1/6 | pybel-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-pybel-conversion-openbabel-before-use | 3.10 | 安装包 | 1/6 | pybel-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-rd-conversion-rdkit-before-hotpot | 3.10 | 安装包 | 1/6 | rd-conversion；屏蔽 rdkit；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-rd-conversion-rdkit-before-use | 3.10 | 安装包 | 1/6 | rd-conversion；屏蔽 rdkit；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-io-read-openbabel-before-hotpot | 3.10 | 安装包 | 5/6 | io-read；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-io-read-openbabel-before-use | 3.10 | 安装包 | 5/6 | io-read；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-io-write-openbabel-before-hotpot | 3.10 | 安装包 | 5/6 | io-write；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-io-write-openbabel-before-use | 3.10 | 安装包 | 5/6 | io-write；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-crystal-cell-openbabel-before-hotpot | 3.10 | 安装包 | 1/6 | crystal-cell；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-crystal-cell-openbabel-before-use | 3.10 | 安装包 | 1/6 | crystal-cell；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp310-installed-missing-thermo-thermo-before-hotpot | 3.10 | 安装包 | 7 | thermo；屏蔽 thermo；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-missing-thermo-thermo-before-use | 3.10 | 安装包 | 7 | thermo；屏蔽 thermo；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-missing-thermo-pandas-before-hotpot | 3.10 | 安装包 | 7 | thermo；屏蔽 pandas；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-missing-thermo-pandas-before-use | 3.10 | 安装包 | 7 | thermo；屏蔽 pandas；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-svg-cairosvg-before-hotpot | 3.10 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-svg-cairosvg-before-use | 3.10 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-svg-matplotlib-before-hotpot | 3.10 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-svg-matplotlib-before-use | 3.10 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-png-cairosvg-before-hotpot | 3.10 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-png-cairosvg-before-use | 3.10 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-webp-PIL-before-hotpot | 3.10 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-webp-PIL-before-use | 3.10 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-use；基础夹具 | 故障注入/第三方边界 |
| cp310-installed-prior-cold-unpickle-0 | 3.10 | 安装包 | 1/5 | molecule-0.pickle | 持久化/事实恢复 |
| cp310-installed-prior-cold-unpickle-1 | 3.10 | 安装包 | 1/5 | molecule-1.pickle | 持久化/事实恢复 |
| cp310-installed-prior-cold-unpickle-2 | 3.10 | 安装包 | 1/5 | molecule-2.pickle | 持久化/事实恢复 |
| cp310-installed-prior-cold-unpickle-3 | 3.10 | 安装包 | 1/5 | molecule-3.pickle | 持久化/事实恢复 |
| cp310-installed-prior-cold-unpickle-4 | 3.10 | 安装包 | 1/5 | molecule-4.pickle | 持久化/事实恢复 |
| cp310-installed-prior-cold-unpickle-5 | 3.10 | 安装包 | 1/5 | molecule-5.pickle | 持久化/事实恢复 |
| cp310-installed-p04-prior-cell-cold-unpickle | 3.10 | 安装包 | 1/5 | c61-cell-molecule.pickle | 持久化/事实恢复 |
| cp310-installed-complete-reflection | 3.10 | 安装包 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |
| cp310-installed-preserve-surface | 3.10 | 安装包 | 1/6（接口） | surface | 接口兼容；非计算流程 |
| cp310-installed-preserve-conversions | 3.10 | 安装包 | 1/2/6 | conversions | 结构运输小样本；非全规则验证 |
| cp310-installed-preserve-io | 3.10 | 安装包 | 1/5/6 | io | I/O小样本；非全格式验证 |
| cp310-installed-preserve-crystal | 3.10 | 安装包 | 1/6 | crystal | 晶胞运输小样本 |
| cp310-installed-preserve-thermo | 3.10 | 安装包 | 7 | thermo | 替身接口合约 |
| cp310-installed-preserve-drawing | 3.10 | 安装包 | 7 | drawing | 第三方输出/资源 |
| cp310-installed-io-path-string-smi-single | 3.10 | 安装包 | 1/5/6 | smi-single.smi；Path/字符串入口 | I/O读回小样本 |
| cp310-installed-io-path-string-smi-multi | 3.10 | 安装包 | 1/5/6 | smi-multi.smi；Path/字符串入口 | I/O读回小样本 |
| cp310-installed-io-path-string-mol2-single | 3.10 | 安装包 | 1/5/6 | mol2-single.mol2；Path/字符串入口 | I/O读回小样本 |
| cp310-installed-io-path-string-mol2-multi | 3.10 | 安装包 | 1/5/6 | mol2-multi.mol2；Path/字符串入口 | I/O读回小样本 |
| cp310-installed-io-path-string-sdf-single | 3.10 | 安装包 | 1/5/6 | sdf-single.sdf；Path/字符串入口 | I/O读回小样本 |
| cp310-installed-io-path-string-sdf-multi | 3.10 | 安装包 | 1/5/6 | sdf-multi.sdf；Path/字符串入口 | I/O读回小样本 |
| cp310-installed-drawing-independent-cases | 3.10 | 安装包 | 7 | 字体、grid、single-string、版本拒绝顺序、10种绘图格式 | 第三方输出/资源 |
| cp311-source-cold-import-00 | 3.11 | 源码 | 1/2/5/6（间接） | hotpot | 导入/依赖边界 |
| cp311-source-cold-import-01 | 3.11 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.structure | 导入/依赖边界 |
| cp311-source-cold-import-02 | 3.11 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.structure.crystal | 导入/依赖边界 |
| cp311-source-cold-import-03 | 3.11 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.operations.conversion | 导入/依赖边界 |
| cp311-source-cold-import-04 | 3.11 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.conversion | 导入/依赖边界 |
| cp311-source-cold-import-05 | 3.11 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.conversion | 导入/依赖边界 |
| cp311-source-cold-import-06 | 3.11 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.crystal | 导入/依赖边界 |
| cp311-source-cold-import-07 | 3.11 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.conversion | 导入/依赖边界 |
| cp311-source-cold-import-08 | 3.11 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.stereo_transport | 导入/依赖边界 |
| cp311-source-cold-import-09 | 3.11 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.operations.io | 导入/依赖边界 |
| cp311-source-cold-import-10 | 3.11 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.io.formats.openbabel | 导入/依赖边界 |
| cp311-source-cold-import-11 | 3.11 | 源码 | 7 | hotpot.cheminfo.operations.thermo | 导入/依赖边界 |
| cp311-source-cold-import-12 | 3.11 | 源码 | 7 | hotpot.cheminfo.adapters.thermo.chemical | 导入/依赖边界 |
| cp311-source-cold-import-13 | 3.11 | 源码 | 7 | hotpot.cheminfo.operations.visualization | 导入/依赖边界 |
| cp311-source-cold-import-14 | 3.11 | 源码 | 7 | hotpot.cheminfo.adapters.visualization.draw | 导入/依赖边界 |
| cp311-source-adapter-order-ob-then-rd | 3.11 | 源码 | 1/6 | hotpot；顺序 ob-then-rd | 小样本转换 |
| cp311-source-adapter-order-rd-then-ob | 3.11 | 源码 | 1/6 | hotpot；顺序 rd-then-ob | 小样本转换 |
| cp311-source-missing-ob-conversion-openbabel-before-hotpot | 3.11 | 源码 | 1/6 | ob-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-ob-conversion-openbabel-before-use | 3.11 | 源码 | 1/6 | ob-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-pybel-conversion-openbabel-before-hotpot | 3.11 | 源码 | 1/6 | pybel-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-pybel-conversion-openbabel-before-use | 3.11 | 源码 | 1/6 | pybel-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-rd-conversion-rdkit-before-hotpot | 3.11 | 源码 | 1/6 | rd-conversion；屏蔽 rdkit；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-rd-conversion-rdkit-before-use | 3.11 | 源码 | 1/6 | rd-conversion；屏蔽 rdkit；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-io-read-openbabel-before-hotpot | 3.11 | 源码 | 5/6 | io-read；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-io-read-openbabel-before-use | 3.11 | 源码 | 5/6 | io-read；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-io-write-openbabel-before-hotpot | 3.11 | 源码 | 5/6 | io-write；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-io-write-openbabel-before-use | 3.11 | 源码 | 5/6 | io-write；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-crystal-cell-openbabel-before-hotpot | 3.11 | 源码 | 1/6 | crystal-cell；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-crystal-cell-openbabel-before-use | 3.11 | 源码 | 1/6 | crystal-cell；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-source-missing-thermo-thermo-before-hotpot | 3.11 | 源码 | 7 | thermo；屏蔽 thermo；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-source-missing-thermo-thermo-before-use | 3.11 | 源码 | 7 | thermo；屏蔽 thermo；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-source-missing-thermo-pandas-before-hotpot | 3.11 | 源码 | 7 | thermo；屏蔽 pandas；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-source-missing-thermo-pandas-before-use | 3.11 | 源码 | 7 | thermo；屏蔽 pandas；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-source-missing-draw-svg-cairosvg-before-hotpot | 3.11 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-source-missing-draw-svg-cairosvg-before-use | 3.11 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-source-missing-draw-svg-matplotlib-before-hotpot | 3.11 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-source-missing-draw-svg-matplotlib-before-use | 3.11 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-source-missing-draw-png-cairosvg-before-hotpot | 3.11 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-source-missing-draw-png-cairosvg-before-use | 3.11 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-source-missing-draw-webp-PIL-before-hotpot | 3.11 | 源码 | 7 | draw-webp；屏蔽 PIL；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-source-missing-draw-webp-PIL-before-use | 3.11 | 源码 | 7 | draw-webp；屏蔽 PIL；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-source-prior-cold-unpickle-0 | 3.11 | 源码 | 1/5 | molecule-0.pickle | 持久化/事实恢复 |
| cp311-source-prior-cold-unpickle-1 | 3.11 | 源码 | 1/5 | molecule-1.pickle | 持久化/事实恢复 |
| cp311-source-prior-cold-unpickle-2 | 3.11 | 源码 | 1/5 | molecule-2.pickle | 持久化/事实恢复 |
| cp311-source-prior-cold-unpickle-3 | 3.11 | 源码 | 1/5 | molecule-3.pickle | 持久化/事实恢复 |
| cp311-source-prior-cold-unpickle-4 | 3.11 | 源码 | 1/5 | molecule-4.pickle | 持久化/事实恢复 |
| cp311-source-prior-cold-unpickle-5 | 3.11 | 源码 | 1/5 | molecule-5.pickle | 持久化/事实恢复 |
| cp311-source-p04-prior-cell-cold-unpickle | 3.11 | 源码 | 1/5 | c61-cell-molecule.pickle | 持久化/事实恢复 |
| cp311-source-complete-reflection | 3.11 | 源码 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |
| cp311-source-preserve-surface | 3.11 | 源码 | 1/6（接口） | surface | 接口兼容；非计算流程 |
| cp311-source-preserve-conversions | 3.11 | 源码 | 1/2/6 | conversions | 结构运输小样本；非全规则验证 |
| cp311-source-preserve-io | 3.11 | 源码 | 1/5/6 | io | I/O小样本；非全格式验证 |
| cp311-source-preserve-crystal | 3.11 | 源码 | 1/6 | crystal | 晶胞运输小样本 |
| cp311-source-preserve-thermo | 3.11 | 源码 | 7 | thermo | 替身接口合约 |
| cp311-source-preserve-drawing | 3.11 | 源码 | 7 | drawing | 第三方输出/资源 |
| cp311-source-io-path-string-smi-single | 3.11 | 源码 | 1/5/6 | smi-single.smi；Path/字符串入口 | I/O读回小样本 |
| cp311-source-io-path-string-smi-multi | 3.11 | 源码 | 1/5/6 | smi-multi.smi；Path/字符串入口 | I/O读回小样本 |
| cp311-source-io-path-string-mol2-single | 3.11 | 源码 | 1/5/6 | mol2-single.mol2；Path/字符串入口 | I/O读回小样本 |
| cp311-source-io-path-string-mol2-multi | 3.11 | 源码 | 1/5/6 | mol2-multi.mol2；Path/字符串入口 | I/O读回小样本 |
| cp311-source-io-path-string-sdf-single | 3.11 | 源码 | 1/5/6 | sdf-single.sdf；Path/字符串入口 | I/O读回小样本 |
| cp311-source-io-path-string-sdf-multi | 3.11 | 源码 | 1/5/6 | sdf-multi.sdf；Path/字符串入口 | I/O读回小样本 |
| cp311-source-drawing-independent-cases | 3.11 | 源码 | 7 | 字体、grid、single-string、版本拒绝顺序、10种绘图格式 | 第三方输出/资源 |
| cp311-installed-cold-import-00 | 3.11 | 安装包 | 1/2/5/6（间接） | hotpot | 导入/依赖边界 |
| cp311-installed-cold-import-01 | 3.11 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.structure | 导入/依赖边界 |
| cp311-installed-cold-import-02 | 3.11 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.structure.crystal | 导入/依赖边界 |
| cp311-installed-cold-import-03 | 3.11 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.operations.conversion | 导入/依赖边界 |
| cp311-installed-cold-import-04 | 3.11 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.conversion | 导入/依赖边界 |
| cp311-installed-cold-import-05 | 3.11 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.conversion | 导入/依赖边界 |
| cp311-installed-cold-import-06 | 3.11 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.crystal | 导入/依赖边界 |
| cp311-installed-cold-import-07 | 3.11 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.conversion | 导入/依赖边界 |
| cp311-installed-cold-import-08 | 3.11 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.stereo_transport | 导入/依赖边界 |
| cp311-installed-cold-import-09 | 3.11 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.operations.io | 导入/依赖边界 |
| cp311-installed-cold-import-10 | 3.11 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.io.formats.openbabel | 导入/依赖边界 |
| cp311-installed-cold-import-11 | 3.11 | 安装包 | 7 | hotpot.cheminfo.operations.thermo | 导入/依赖边界 |
| cp311-installed-cold-import-12 | 3.11 | 安装包 | 7 | hotpot.cheminfo.adapters.thermo.chemical | 导入/依赖边界 |
| cp311-installed-cold-import-13 | 3.11 | 安装包 | 7 | hotpot.cheminfo.operations.visualization | 导入/依赖边界 |
| cp311-installed-cold-import-14 | 3.11 | 安装包 | 7 | hotpot.cheminfo.adapters.visualization.draw | 导入/依赖边界 |
| cp311-installed-adapter-order-ob-then-rd | 3.11 | 安装包 | 1/6 | hotpot；顺序 ob-then-rd | 小样本转换 |
| cp311-installed-adapter-order-rd-then-ob | 3.11 | 安装包 | 1/6 | hotpot；顺序 rd-then-ob | 小样本转换 |
| cp311-installed-missing-ob-conversion-openbabel-before-hotpot | 3.11 | 安装包 | 1/6 | ob-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-ob-conversion-openbabel-before-use | 3.11 | 安装包 | 1/6 | ob-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-pybel-conversion-openbabel-before-hotpot | 3.11 | 安装包 | 1/6 | pybel-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-pybel-conversion-openbabel-before-use | 3.11 | 安装包 | 1/6 | pybel-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-rd-conversion-rdkit-before-hotpot | 3.11 | 安装包 | 1/6 | rd-conversion；屏蔽 rdkit；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-rd-conversion-rdkit-before-use | 3.11 | 安装包 | 1/6 | rd-conversion；屏蔽 rdkit；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-io-read-openbabel-before-hotpot | 3.11 | 安装包 | 5/6 | io-read；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-io-read-openbabel-before-use | 3.11 | 安装包 | 5/6 | io-read；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-io-write-openbabel-before-hotpot | 3.11 | 安装包 | 5/6 | io-write；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-io-write-openbabel-before-use | 3.11 | 安装包 | 5/6 | io-write；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-crystal-cell-openbabel-before-hotpot | 3.11 | 安装包 | 1/6 | crystal-cell；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-crystal-cell-openbabel-before-use | 3.11 | 安装包 | 1/6 | crystal-cell；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp311-installed-missing-thermo-thermo-before-hotpot | 3.11 | 安装包 | 7 | thermo；屏蔽 thermo；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-missing-thermo-thermo-before-use | 3.11 | 安装包 | 7 | thermo；屏蔽 thermo；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-missing-thermo-pandas-before-hotpot | 3.11 | 安装包 | 7 | thermo；屏蔽 pandas；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-missing-thermo-pandas-before-use | 3.11 | 安装包 | 7 | thermo；屏蔽 pandas；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-svg-cairosvg-before-hotpot | 3.11 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-svg-cairosvg-before-use | 3.11 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-svg-matplotlib-before-hotpot | 3.11 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-svg-matplotlib-before-use | 3.11 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-png-cairosvg-before-hotpot | 3.11 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-png-cairosvg-before-use | 3.11 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-webp-PIL-before-hotpot | 3.11 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-webp-PIL-before-use | 3.11 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-use；基础夹具 | 故障注入/第三方边界 |
| cp311-installed-prior-cold-unpickle-0 | 3.11 | 安装包 | 1/5 | molecule-0.pickle | 持久化/事实恢复 |
| cp311-installed-prior-cold-unpickle-1 | 3.11 | 安装包 | 1/5 | molecule-1.pickle | 持久化/事实恢复 |
| cp311-installed-prior-cold-unpickle-2 | 3.11 | 安装包 | 1/5 | molecule-2.pickle | 持久化/事实恢复 |
| cp311-installed-prior-cold-unpickle-3 | 3.11 | 安装包 | 1/5 | molecule-3.pickle | 持久化/事实恢复 |
| cp311-installed-prior-cold-unpickle-4 | 3.11 | 安装包 | 1/5 | molecule-4.pickle | 持久化/事实恢复 |
| cp311-installed-prior-cold-unpickle-5 | 3.11 | 安装包 | 1/5 | molecule-5.pickle | 持久化/事实恢复 |
| cp311-installed-p04-prior-cell-cold-unpickle | 3.11 | 安装包 | 1/5 | c61-cell-molecule.pickle | 持久化/事实恢复 |
| cp311-installed-complete-reflection | 3.11 | 安装包 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |
| cp311-installed-preserve-surface | 3.11 | 安装包 | 1/6（接口） | surface | 接口兼容；非计算流程 |
| cp311-installed-preserve-conversions | 3.11 | 安装包 | 1/2/6 | conversions | 结构运输小样本；非全规则验证 |
| cp311-installed-preserve-io | 3.11 | 安装包 | 1/5/6 | io | I/O小样本；非全格式验证 |
| cp311-installed-preserve-crystal | 3.11 | 安装包 | 1/6 | crystal | 晶胞运输小样本 |
| cp311-installed-preserve-thermo | 3.11 | 安装包 | 7 | thermo | 替身接口合约 |
| cp311-installed-preserve-drawing | 3.11 | 安装包 | 7 | drawing | 第三方输出/资源 |
| cp311-installed-io-path-string-smi-single | 3.11 | 安装包 | 1/5/6 | smi-single.smi；Path/字符串入口 | I/O读回小样本 |
| cp311-installed-io-path-string-smi-multi | 3.11 | 安装包 | 1/5/6 | smi-multi.smi；Path/字符串入口 | I/O读回小样本 |
| cp311-installed-io-path-string-mol2-single | 3.11 | 安装包 | 1/5/6 | mol2-single.mol2；Path/字符串入口 | I/O读回小样本 |
| cp311-installed-io-path-string-mol2-multi | 3.11 | 安装包 | 1/5/6 | mol2-multi.mol2；Path/字符串入口 | I/O读回小样本 |
| cp311-installed-io-path-string-sdf-single | 3.11 | 安装包 | 1/5/6 | sdf-single.sdf；Path/字符串入口 | I/O读回小样本 |
| cp311-installed-io-path-string-sdf-multi | 3.11 | 安装包 | 1/5/6 | sdf-multi.sdf；Path/字符串入口 | I/O读回小样本 |
| cp311-installed-drawing-independent-cases | 3.11 | 安装包 | 7 | 字体、grid、single-string、版本拒绝顺序、10种绘图格式 | 第三方输出/资源 |
| cp312-source-cold-import-00 | 3.12 | 源码 | 1/2/5/6（间接） | hotpot | 导入/依赖边界 |
| cp312-source-cold-import-01 | 3.12 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.structure | 导入/依赖边界 |
| cp312-source-cold-import-02 | 3.12 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.structure.crystal | 导入/依赖边界 |
| cp312-source-cold-import-03 | 3.12 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.operations.conversion | 导入/依赖边界 |
| cp312-source-cold-import-04 | 3.12 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.conversion | 导入/依赖边界 |
| cp312-source-cold-import-05 | 3.12 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.conversion | 导入/依赖边界 |
| cp312-source-cold-import-06 | 3.12 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.crystal | 导入/依赖边界 |
| cp312-source-cold-import-07 | 3.12 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.conversion | 导入/依赖边界 |
| cp312-source-cold-import-08 | 3.12 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.stereo_transport | 导入/依赖边界 |
| cp312-source-cold-import-09 | 3.12 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.operations.io | 导入/依赖边界 |
| cp312-source-cold-import-10 | 3.12 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.io.formats.openbabel | 导入/依赖边界 |
| cp312-source-cold-import-11 | 3.12 | 源码 | 7 | hotpot.cheminfo.operations.thermo | 导入/依赖边界 |
| cp312-source-cold-import-12 | 3.12 | 源码 | 7 | hotpot.cheminfo.adapters.thermo.chemical | 导入/依赖边界 |
| cp312-source-cold-import-13 | 3.12 | 源码 | 7 | hotpot.cheminfo.operations.visualization | 导入/依赖边界 |
| cp312-source-cold-import-14 | 3.12 | 源码 | 7 | hotpot.cheminfo.adapters.visualization.draw | 导入/依赖边界 |
| cp312-source-adapter-order-ob-then-rd | 3.12 | 源码 | 1/6 | hotpot；顺序 ob-then-rd | 小样本转换 |
| cp312-source-adapter-order-rd-then-ob | 3.12 | 源码 | 1/6 | hotpot；顺序 rd-then-ob | 小样本转换 |
| cp312-source-missing-ob-conversion-openbabel-before-hotpot | 3.12 | 源码 | 1/6 | ob-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-ob-conversion-openbabel-before-use | 3.12 | 源码 | 1/6 | ob-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-pybel-conversion-openbabel-before-hotpot | 3.12 | 源码 | 1/6 | pybel-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-pybel-conversion-openbabel-before-use | 3.12 | 源码 | 1/6 | pybel-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-rd-conversion-rdkit-before-hotpot | 3.12 | 源码 | 1/6 | rd-conversion；屏蔽 rdkit；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-rd-conversion-rdkit-before-use | 3.12 | 源码 | 1/6 | rd-conversion；屏蔽 rdkit；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-io-read-openbabel-before-hotpot | 3.12 | 源码 | 5/6 | io-read；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-io-read-openbabel-before-use | 3.12 | 源码 | 5/6 | io-read；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-io-write-openbabel-before-hotpot | 3.12 | 源码 | 5/6 | io-write；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-io-write-openbabel-before-use | 3.12 | 源码 | 5/6 | io-write；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-crystal-cell-openbabel-before-hotpot | 3.12 | 源码 | 1/6 | crystal-cell；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-crystal-cell-openbabel-before-use | 3.12 | 源码 | 1/6 | crystal-cell；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-source-missing-thermo-thermo-before-hotpot | 3.12 | 源码 | 7 | thermo；屏蔽 thermo；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-source-missing-thermo-thermo-before-use | 3.12 | 源码 | 7 | thermo；屏蔽 thermo；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-source-missing-thermo-pandas-before-hotpot | 3.12 | 源码 | 7 | thermo；屏蔽 pandas；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-source-missing-thermo-pandas-before-use | 3.12 | 源码 | 7 | thermo；屏蔽 pandas；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-source-missing-draw-svg-cairosvg-before-hotpot | 3.12 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-source-missing-draw-svg-cairosvg-before-use | 3.12 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-source-missing-draw-svg-matplotlib-before-hotpot | 3.12 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-source-missing-draw-svg-matplotlib-before-use | 3.12 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-source-missing-draw-png-cairosvg-before-hotpot | 3.12 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-source-missing-draw-png-cairosvg-before-use | 3.12 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-source-missing-draw-webp-PIL-before-hotpot | 3.12 | 源码 | 7 | draw-webp；屏蔽 PIL；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-source-missing-draw-webp-PIL-before-use | 3.12 | 源码 | 7 | draw-webp；屏蔽 PIL；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-source-prior-cold-unpickle-0 | 3.12 | 源码 | 1/5 | molecule-0.pickle | 持久化/事实恢复 |
| cp312-source-prior-cold-unpickle-1 | 3.12 | 源码 | 1/5 | molecule-1.pickle | 持久化/事实恢复 |
| cp312-source-prior-cold-unpickle-2 | 3.12 | 源码 | 1/5 | molecule-2.pickle | 持久化/事实恢复 |
| cp312-source-prior-cold-unpickle-3 | 3.12 | 源码 | 1/5 | molecule-3.pickle | 持久化/事实恢复 |
| cp312-source-prior-cold-unpickle-4 | 3.12 | 源码 | 1/5 | molecule-4.pickle | 持久化/事实恢复 |
| cp312-source-prior-cold-unpickle-5 | 3.12 | 源码 | 1/5 | molecule-5.pickle | 持久化/事实恢复 |
| cp312-source-p04-prior-cell-cold-unpickle | 3.12 | 源码 | 1/5 | c61-cell-molecule.pickle | 持久化/事实恢复 |
| cp312-source-complete-reflection | 3.12 | 源码 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |
| cp312-source-preserve-surface | 3.12 | 源码 | 1/6（接口） | surface | 接口兼容；非计算流程 |
| cp312-source-preserve-conversions | 3.12 | 源码 | 1/2/6 | conversions | 结构运输小样本；非全规则验证 |
| cp312-source-preserve-io | 3.12 | 源码 | 1/5/6 | io | I/O小样本；非全格式验证 |
| cp312-source-preserve-crystal | 3.12 | 源码 | 1/6 | crystal | 晶胞运输小样本 |
| cp312-source-preserve-thermo | 3.12 | 源码 | 7 | thermo | 替身接口合约 |
| cp312-source-preserve-drawing | 3.12 | 源码 | 7 | drawing | 第三方输出/资源 |
| cp312-source-io-path-string-smi-single | 3.12 | 源码 | 1/5/6 | smi-single.smi；Path/字符串入口 | I/O读回小样本 |
| cp312-source-io-path-string-smi-multi | 3.12 | 源码 | 1/5/6 | smi-multi.smi；Path/字符串入口 | I/O读回小样本 |
| cp312-source-io-path-string-mol2-single | 3.12 | 源码 | 1/5/6 | mol2-single.mol2；Path/字符串入口 | I/O读回小样本 |
| cp312-source-io-path-string-mol2-multi | 3.12 | 源码 | 1/5/6 | mol2-multi.mol2；Path/字符串入口 | I/O读回小样本 |
| cp312-source-io-path-string-sdf-single | 3.12 | 源码 | 1/5/6 | sdf-single.sdf；Path/字符串入口 | I/O读回小样本 |
| cp312-source-io-path-string-sdf-multi | 3.12 | 源码 | 1/5/6 | sdf-multi.sdf；Path/字符串入口 | I/O读回小样本 |
| cp312-source-drawing-independent-cases | 3.12 | 源码 | 7 | 字体、grid、single-string、版本拒绝顺序、10种绘图格式 | 第三方输出/资源 |
| cp312-installed-cold-import-00 | 3.12 | 安装包 | 1/2/5/6（间接） | hotpot | 导入/依赖边界 |
| cp312-installed-cold-import-01 | 3.12 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.structure | 导入/依赖边界 |
| cp312-installed-cold-import-02 | 3.12 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.structure.crystal | 导入/依赖边界 |
| cp312-installed-cold-import-03 | 3.12 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.operations.conversion | 导入/依赖边界 |
| cp312-installed-cold-import-04 | 3.12 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.conversion | 导入/依赖边界 |
| cp312-installed-cold-import-05 | 3.12 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.conversion | 导入/依赖边界 |
| cp312-installed-cold-import-06 | 3.12 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.crystal | 导入/依赖边界 |
| cp312-installed-cold-import-07 | 3.12 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.conversion | 导入/依赖边界 |
| cp312-installed-cold-import-08 | 3.12 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.stereo_transport | 导入/依赖边界 |
| cp312-installed-cold-import-09 | 3.12 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.operations.io | 导入/依赖边界 |
| cp312-installed-cold-import-10 | 3.12 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.io.formats.openbabel | 导入/依赖边界 |
| cp312-installed-cold-import-11 | 3.12 | 安装包 | 7 | hotpot.cheminfo.operations.thermo | 导入/依赖边界 |
| cp312-installed-cold-import-12 | 3.12 | 安装包 | 7 | hotpot.cheminfo.adapters.thermo.chemical | 导入/依赖边界 |
| cp312-installed-cold-import-13 | 3.12 | 安装包 | 7 | hotpot.cheminfo.operations.visualization | 导入/依赖边界 |
| cp312-installed-cold-import-14 | 3.12 | 安装包 | 7 | hotpot.cheminfo.adapters.visualization.draw | 导入/依赖边界 |
| cp312-installed-adapter-order-ob-then-rd | 3.12 | 安装包 | 1/6 | hotpot；顺序 ob-then-rd | 小样本转换 |
| cp312-installed-adapter-order-rd-then-ob | 3.12 | 安装包 | 1/6 | hotpot；顺序 rd-then-ob | 小样本转换 |
| cp312-installed-missing-ob-conversion-openbabel-before-hotpot | 3.12 | 安装包 | 1/6 | ob-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-ob-conversion-openbabel-before-use | 3.12 | 安装包 | 1/6 | ob-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-pybel-conversion-openbabel-before-hotpot | 3.12 | 安装包 | 1/6 | pybel-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-pybel-conversion-openbabel-before-use | 3.12 | 安装包 | 1/6 | pybel-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-rd-conversion-rdkit-before-hotpot | 3.12 | 安装包 | 1/6 | rd-conversion；屏蔽 rdkit；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-rd-conversion-rdkit-before-use | 3.12 | 安装包 | 1/6 | rd-conversion；屏蔽 rdkit；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-io-read-openbabel-before-hotpot | 3.12 | 安装包 | 5/6 | io-read；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-io-read-openbabel-before-use | 3.12 | 安装包 | 5/6 | io-read；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-io-write-openbabel-before-hotpot | 3.12 | 安装包 | 5/6 | io-write；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-io-write-openbabel-before-use | 3.12 | 安装包 | 5/6 | io-write；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-crystal-cell-openbabel-before-hotpot | 3.12 | 安装包 | 1/6 | crystal-cell；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-crystal-cell-openbabel-before-use | 3.12 | 安装包 | 1/6 | crystal-cell；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp312-installed-missing-thermo-thermo-before-hotpot | 3.12 | 安装包 | 7 | thermo；屏蔽 thermo；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-missing-thermo-thermo-before-use | 3.12 | 安装包 | 7 | thermo；屏蔽 thermo；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-missing-thermo-pandas-before-hotpot | 3.12 | 安装包 | 7 | thermo；屏蔽 pandas；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-missing-thermo-pandas-before-use | 3.12 | 安装包 | 7 | thermo；屏蔽 pandas；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-svg-cairosvg-before-hotpot | 3.12 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-svg-cairosvg-before-use | 3.12 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-svg-matplotlib-before-hotpot | 3.12 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-svg-matplotlib-before-use | 3.12 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-png-cairosvg-before-hotpot | 3.12 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-png-cairosvg-before-use | 3.12 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-webp-PIL-before-hotpot | 3.12 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-webp-PIL-before-use | 3.12 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-use；基础夹具 | 故障注入/第三方边界 |
| cp312-installed-prior-cold-unpickle-0 | 3.12 | 安装包 | 1/5 | molecule-0.pickle | 持久化/事实恢复 |
| cp312-installed-prior-cold-unpickle-1 | 3.12 | 安装包 | 1/5 | molecule-1.pickle | 持久化/事实恢复 |
| cp312-installed-prior-cold-unpickle-2 | 3.12 | 安装包 | 1/5 | molecule-2.pickle | 持久化/事实恢复 |
| cp312-installed-prior-cold-unpickle-3 | 3.12 | 安装包 | 1/5 | molecule-3.pickle | 持久化/事实恢复 |
| cp312-installed-prior-cold-unpickle-4 | 3.12 | 安装包 | 1/5 | molecule-4.pickle | 持久化/事实恢复 |
| cp312-installed-prior-cold-unpickle-5 | 3.12 | 安装包 | 1/5 | molecule-5.pickle | 持久化/事实恢复 |
| cp312-installed-p04-prior-cell-cold-unpickle | 3.12 | 安装包 | 1/5 | c61-cell-molecule.pickle | 持久化/事实恢复 |
| cp312-installed-complete-reflection | 3.12 | 安装包 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |
| cp312-installed-preserve-surface | 3.12 | 安装包 | 1/6（接口） | surface | 接口兼容；非计算流程 |
| cp312-installed-preserve-conversions | 3.12 | 安装包 | 1/2/6 | conversions | 结构运输小样本；非全规则验证 |
| cp312-installed-preserve-io | 3.12 | 安装包 | 1/5/6 | io | I/O小样本；非全格式验证 |
| cp312-installed-preserve-crystal | 3.12 | 安装包 | 1/6 | crystal | 晶胞运输小样本 |
| cp312-installed-preserve-thermo | 3.12 | 安装包 | 7 | thermo | 替身接口合约 |
| cp312-installed-preserve-drawing | 3.12 | 安装包 | 7 | drawing | 第三方输出/资源 |
| cp312-installed-io-path-string-smi-single | 3.12 | 安装包 | 1/5/6 | smi-single.smi；Path/字符串入口 | I/O读回小样本 |
| cp312-installed-io-path-string-smi-multi | 3.12 | 安装包 | 1/5/6 | smi-multi.smi；Path/字符串入口 | I/O读回小样本 |
| cp312-installed-io-path-string-mol2-single | 3.12 | 安装包 | 1/5/6 | mol2-single.mol2；Path/字符串入口 | I/O读回小样本 |
| cp312-installed-io-path-string-mol2-multi | 3.12 | 安装包 | 1/5/6 | mol2-multi.mol2；Path/字符串入口 | I/O读回小样本 |
| cp312-installed-io-path-string-sdf-single | 3.12 | 安装包 | 1/5/6 | sdf-single.sdf；Path/字符串入口 | I/O读回小样本 |
| cp312-installed-io-path-string-sdf-multi | 3.12 | 安装包 | 1/5/6 | sdf-multi.sdf；Path/字符串入口 | I/O读回小样本 |
| cp312-installed-drawing-independent-cases | 3.12 | 安装包 | 7 | 字体、grid、single-string、版本拒绝顺序、10种绘图格式 | 第三方输出/资源 |
| cp313-source-cold-import-00 | 3.13 | 源码 | 1/2/5/6（间接） | hotpot | 导入/依赖边界 |
| cp313-source-cold-import-01 | 3.13 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.structure | 导入/依赖边界 |
| cp313-source-cold-import-02 | 3.13 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.structure.crystal | 导入/依赖边界 |
| cp313-source-cold-import-03 | 3.13 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.operations.conversion | 导入/依赖边界 |
| cp313-source-cold-import-04 | 3.13 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.conversion | 导入/依赖边界 |
| cp313-source-cold-import-05 | 3.13 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.conversion | 导入/依赖边界 |
| cp313-source-cold-import-06 | 3.13 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.crystal | 导入/依赖边界 |
| cp313-source-cold-import-07 | 3.13 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.conversion | 导入/依赖边界 |
| cp313-source-cold-import-08 | 3.13 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.stereo_transport | 导入/依赖边界 |
| cp313-source-cold-import-09 | 3.13 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.operations.io | 导入/依赖边界 |
| cp313-source-cold-import-10 | 3.13 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.io.formats.openbabel | 导入/依赖边界 |
| cp313-source-cold-import-11 | 3.13 | 源码 | 7 | hotpot.cheminfo.operations.thermo | 导入/依赖边界 |
| cp313-source-cold-import-12 | 3.13 | 源码 | 7 | hotpot.cheminfo.adapters.thermo.chemical | 导入/依赖边界 |
| cp313-source-cold-import-13 | 3.13 | 源码 | 7 | hotpot.cheminfo.operations.visualization | 导入/依赖边界 |
| cp313-source-cold-import-14 | 3.13 | 源码 | 7 | hotpot.cheminfo.adapters.visualization.draw | 导入/依赖边界 |
| cp313-source-adapter-order-ob-then-rd | 3.13 | 源码 | 1/6 | hotpot；顺序 ob-then-rd | 小样本转换 |
| cp313-source-adapter-order-rd-then-ob | 3.13 | 源码 | 1/6 | hotpot；顺序 rd-then-ob | 小样本转换 |
| cp313-source-missing-ob-conversion-openbabel-before-hotpot | 3.13 | 源码 | 1/6 | ob-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-ob-conversion-openbabel-before-use | 3.13 | 源码 | 1/6 | ob-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-pybel-conversion-openbabel-before-hotpot | 3.13 | 源码 | 1/6 | pybel-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-pybel-conversion-openbabel-before-use | 3.13 | 源码 | 1/6 | pybel-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-rd-conversion-rdkit-before-hotpot | 3.13 | 源码 | 1/6 | rd-conversion；屏蔽 rdkit；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-rd-conversion-rdkit-before-use | 3.13 | 源码 | 1/6 | rd-conversion；屏蔽 rdkit；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-io-read-openbabel-before-hotpot | 3.13 | 源码 | 5/6 | io-read；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-io-read-openbabel-before-use | 3.13 | 源码 | 5/6 | io-read；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-io-write-openbabel-before-hotpot | 3.13 | 源码 | 5/6 | io-write；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-io-write-openbabel-before-use | 3.13 | 源码 | 5/6 | io-write；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-crystal-cell-openbabel-before-hotpot | 3.13 | 源码 | 1/6 | crystal-cell；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-crystal-cell-openbabel-before-use | 3.13 | 源码 | 1/6 | crystal-cell；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-source-missing-thermo-thermo-before-hotpot | 3.13 | 源码 | 7 | thermo；屏蔽 thermo；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-source-missing-thermo-thermo-before-use | 3.13 | 源码 | 7 | thermo；屏蔽 thermo；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-source-missing-thermo-pandas-before-hotpot | 3.13 | 源码 | 7 | thermo；屏蔽 pandas；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-source-missing-thermo-pandas-before-use | 3.13 | 源码 | 7 | thermo；屏蔽 pandas；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-source-missing-draw-svg-cairosvg-before-hotpot | 3.13 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-source-missing-draw-svg-cairosvg-before-use | 3.13 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-source-missing-draw-svg-matplotlib-before-hotpot | 3.13 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-source-missing-draw-svg-matplotlib-before-use | 3.13 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-source-missing-draw-png-cairosvg-before-hotpot | 3.13 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-source-missing-draw-png-cairosvg-before-use | 3.13 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-source-missing-draw-webp-PIL-before-hotpot | 3.13 | 源码 | 7 | draw-webp；屏蔽 PIL；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-source-missing-draw-webp-PIL-before-use | 3.13 | 源码 | 7 | draw-webp；屏蔽 PIL；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-source-prior-cold-unpickle-0 | 3.13 | 源码 | 1/5 | molecule-0.pickle | 持久化/事实恢复 |
| cp313-source-prior-cold-unpickle-1 | 3.13 | 源码 | 1/5 | molecule-1.pickle | 持久化/事实恢复 |
| cp313-source-prior-cold-unpickle-2 | 3.13 | 源码 | 1/5 | molecule-2.pickle | 持久化/事实恢复 |
| cp313-source-prior-cold-unpickle-3 | 3.13 | 源码 | 1/5 | molecule-3.pickle | 持久化/事实恢复 |
| cp313-source-prior-cold-unpickle-4 | 3.13 | 源码 | 1/5 | molecule-4.pickle | 持久化/事实恢复 |
| cp313-source-prior-cold-unpickle-5 | 3.13 | 源码 | 1/5 | molecule-5.pickle | 持久化/事实恢复 |
| cp313-source-p04-prior-cell-cold-unpickle | 3.13 | 源码 | 1/5 | c61-cell-molecule.pickle | 持久化/事实恢复 |
| cp313-source-complete-reflection | 3.13 | 源码 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |
| cp313-source-preserve-surface | 3.13 | 源码 | 1/6（接口） | surface | 接口兼容；非计算流程 |
| cp313-source-preserve-conversions | 3.13 | 源码 | 1/2/6 | conversions | 结构运输小样本；非全规则验证 |
| cp313-source-preserve-io | 3.13 | 源码 | 1/5/6 | io | I/O小样本；非全格式验证 |
| cp313-source-preserve-crystal | 3.13 | 源码 | 1/6 | crystal | 晶胞运输小样本 |
| cp313-source-preserve-thermo | 3.13 | 源码 | 7 | thermo | 替身接口合约 |
| cp313-source-preserve-drawing | 3.13 | 源码 | 7 | drawing | 第三方输出/资源 |
| cp313-source-io-path-string-smi-single | 3.13 | 源码 | 1/5/6 | smi-single.smi；Path/字符串入口 | I/O读回小样本 |
| cp313-source-io-path-string-smi-multi | 3.13 | 源码 | 1/5/6 | smi-multi.smi；Path/字符串入口 | I/O读回小样本 |
| cp313-source-io-path-string-mol2-single | 3.13 | 源码 | 1/5/6 | mol2-single.mol2；Path/字符串入口 | I/O读回小样本 |
| cp313-source-io-path-string-mol2-multi | 3.13 | 源码 | 1/5/6 | mol2-multi.mol2；Path/字符串入口 | I/O读回小样本 |
| cp313-source-io-path-string-sdf-single | 3.13 | 源码 | 1/5/6 | sdf-single.sdf；Path/字符串入口 | I/O读回小样本 |
| cp313-source-io-path-string-sdf-multi | 3.13 | 源码 | 1/5/6 | sdf-multi.sdf；Path/字符串入口 | I/O读回小样本 |
| cp313-source-drawing-independent-cases | 3.13 | 源码 | 7 | 字体、grid、single-string、版本拒绝顺序、10种绘图格式 | 第三方输出/资源 |
| cp313-installed-cold-import-00 | 3.13 | 安装包 | 1/2/5/6（间接） | hotpot | 导入/依赖边界 |
| cp313-installed-cold-import-01 | 3.13 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.structure | 导入/依赖边界 |
| cp313-installed-cold-import-02 | 3.13 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.structure.crystal | 导入/依赖边界 |
| cp313-installed-cold-import-03 | 3.13 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.operations.conversion | 导入/依赖边界 |
| cp313-installed-cold-import-04 | 3.13 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.conversion | 导入/依赖边界 |
| cp313-installed-cold-import-05 | 3.13 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.conversion | 导入/依赖边界 |
| cp313-installed-cold-import-06 | 3.13 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.crystal | 导入/依赖边界 |
| cp313-installed-cold-import-07 | 3.13 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.conversion | 导入/依赖边界 |
| cp313-installed-cold-import-08 | 3.13 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.stereo_transport | 导入/依赖边界 |
| cp313-installed-cold-import-09 | 3.13 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.operations.io | 导入/依赖边界 |
| cp313-installed-cold-import-10 | 3.13 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.io.formats.openbabel | 导入/依赖边界 |
| cp313-installed-cold-import-11 | 3.13 | 安装包 | 7 | hotpot.cheminfo.operations.thermo | 导入/依赖边界 |
| cp313-installed-cold-import-12 | 3.13 | 安装包 | 7 | hotpot.cheminfo.adapters.thermo.chemical | 导入/依赖边界 |
| cp313-installed-cold-import-13 | 3.13 | 安装包 | 7 | hotpot.cheminfo.operations.visualization | 导入/依赖边界 |
| cp313-installed-cold-import-14 | 3.13 | 安装包 | 7 | hotpot.cheminfo.adapters.visualization.draw | 导入/依赖边界 |
| cp313-installed-adapter-order-ob-then-rd | 3.13 | 安装包 | 1/6 | hotpot；顺序 ob-then-rd | 小样本转换 |
| cp313-installed-adapter-order-rd-then-ob | 3.13 | 安装包 | 1/6 | hotpot；顺序 rd-then-ob | 小样本转换 |
| cp313-installed-missing-ob-conversion-openbabel-before-hotpot | 3.13 | 安装包 | 1/6 | ob-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-ob-conversion-openbabel-before-use | 3.13 | 安装包 | 1/6 | ob-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-pybel-conversion-openbabel-before-hotpot | 3.13 | 安装包 | 1/6 | pybel-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-pybel-conversion-openbabel-before-use | 3.13 | 安装包 | 1/6 | pybel-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-rd-conversion-rdkit-before-hotpot | 3.13 | 安装包 | 1/6 | rd-conversion；屏蔽 rdkit；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-rd-conversion-rdkit-before-use | 3.13 | 安装包 | 1/6 | rd-conversion；屏蔽 rdkit；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-io-read-openbabel-before-hotpot | 3.13 | 安装包 | 5/6 | io-read；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-io-read-openbabel-before-use | 3.13 | 安装包 | 5/6 | io-read；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-io-write-openbabel-before-hotpot | 3.13 | 安装包 | 5/6 | io-write；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-io-write-openbabel-before-use | 3.13 | 安装包 | 5/6 | io-write；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-crystal-cell-openbabel-before-hotpot | 3.13 | 安装包 | 1/6 | crystal-cell；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-crystal-cell-openbabel-before-use | 3.13 | 安装包 | 1/6 | crystal-cell；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp313-installed-missing-thermo-thermo-before-hotpot | 3.13 | 安装包 | 7 | thermo；屏蔽 thermo；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-missing-thermo-thermo-before-use | 3.13 | 安装包 | 7 | thermo；屏蔽 thermo；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-missing-thermo-pandas-before-hotpot | 3.13 | 安装包 | 7 | thermo；屏蔽 pandas；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-missing-thermo-pandas-before-use | 3.13 | 安装包 | 7 | thermo；屏蔽 pandas；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-svg-cairosvg-before-hotpot | 3.13 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-svg-cairosvg-before-use | 3.13 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-svg-matplotlib-before-hotpot | 3.13 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-svg-matplotlib-before-use | 3.13 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-png-cairosvg-before-hotpot | 3.13 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-png-cairosvg-before-use | 3.13 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-webp-PIL-before-hotpot | 3.13 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-webp-PIL-before-use | 3.13 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-use；基础夹具 | 故障注入/第三方边界 |
| cp313-installed-prior-cold-unpickle-0 | 3.13 | 安装包 | 1/5 | molecule-0.pickle | 持久化/事实恢复 |
| cp313-installed-prior-cold-unpickle-1 | 3.13 | 安装包 | 1/5 | molecule-1.pickle | 持久化/事实恢复 |
| cp313-installed-prior-cold-unpickle-2 | 3.13 | 安装包 | 1/5 | molecule-2.pickle | 持久化/事实恢复 |
| cp313-installed-prior-cold-unpickle-3 | 3.13 | 安装包 | 1/5 | molecule-3.pickle | 持久化/事实恢复 |
| cp313-installed-prior-cold-unpickle-4 | 3.13 | 安装包 | 1/5 | molecule-4.pickle | 持久化/事实恢复 |
| cp313-installed-prior-cold-unpickle-5 | 3.13 | 安装包 | 1/5 | molecule-5.pickle | 持久化/事实恢复 |
| cp313-installed-p04-prior-cell-cold-unpickle | 3.13 | 安装包 | 1/5 | c61-cell-molecule.pickle | 持久化/事实恢复 |
| cp313-installed-complete-reflection | 3.13 | 安装包 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |
| cp313-installed-preserve-surface | 3.13 | 安装包 | 1/6（接口） | surface | 接口兼容；非计算流程 |
| cp313-installed-preserve-conversions | 3.13 | 安装包 | 1/2/6 | conversions | 结构运输小样本；非全规则验证 |
| cp313-installed-preserve-io | 3.13 | 安装包 | 1/5/6 | io | I/O小样本；非全格式验证 |
| cp313-installed-preserve-crystal | 3.13 | 安装包 | 1/6 | crystal | 晶胞运输小样本 |
| cp313-installed-preserve-thermo | 3.13 | 安装包 | 7 | thermo | 替身接口合约 |
| cp313-installed-preserve-drawing | 3.13 | 安装包 | 7 | drawing | 第三方输出/资源 |
| cp313-installed-io-path-string-smi-single | 3.13 | 安装包 | 1/5/6 | smi-single.smi；Path/字符串入口 | I/O读回小样本 |
| cp313-installed-io-path-string-smi-multi | 3.13 | 安装包 | 1/5/6 | smi-multi.smi；Path/字符串入口 | I/O读回小样本 |
| cp313-installed-io-path-string-mol2-single | 3.13 | 安装包 | 1/5/6 | mol2-single.mol2；Path/字符串入口 | I/O读回小样本 |
| cp313-installed-io-path-string-mol2-multi | 3.13 | 安装包 | 1/5/6 | mol2-multi.mol2；Path/字符串入口 | I/O读回小样本 |
| cp313-installed-io-path-string-sdf-single | 3.13 | 安装包 | 1/5/6 | sdf-single.sdf；Path/字符串入口 | I/O读回小样本 |
| cp313-installed-io-path-string-sdf-multi | 3.13 | 安装包 | 1/5/6 | sdf-multi.sdf；Path/字符串入口 | I/O读回小样本 |
| cp313-installed-drawing-independent-cases | 3.13 | 安装包 | 7 | 字体、grid、single-string、版本拒绝顺序、10种绘图格式 | 第三方输出/资源 |
| cp314-source-cold-import-00 | 3.14 | 源码 | 1/2/5/6（间接） | hotpot | 导入/依赖边界 |
| cp314-source-cold-import-01 | 3.14 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.structure | 导入/依赖边界 |
| cp314-source-cold-import-02 | 3.14 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.structure.crystal | 导入/依赖边界 |
| cp314-source-cold-import-03 | 3.14 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.operations.conversion | 导入/依赖边界 |
| cp314-source-cold-import-04 | 3.14 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.conversion | 导入/依赖边界 |
| cp314-source-cold-import-05 | 3.14 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.conversion | 导入/依赖边界 |
| cp314-source-cold-import-06 | 3.14 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.crystal | 导入/依赖边界 |
| cp314-source-cold-import-07 | 3.14 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.conversion | 导入/依赖边界 |
| cp314-source-cold-import-08 | 3.14 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.stereo_transport | 导入/依赖边界 |
| cp314-source-cold-import-09 | 3.14 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.operations.io | 导入/依赖边界 |
| cp314-source-cold-import-10 | 3.14 | 源码 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.io.formats.openbabel | 导入/依赖边界 |
| cp314-source-cold-import-11 | 3.14 | 源码 | 7 | hotpot.cheminfo.operations.thermo | 导入/依赖边界 |
| cp314-source-cold-import-12 | 3.14 | 源码 | 7 | hotpot.cheminfo.adapters.thermo.chemical | 导入/依赖边界 |
| cp314-source-cold-import-13 | 3.14 | 源码 | 7 | hotpot.cheminfo.operations.visualization | 导入/依赖边界 |
| cp314-source-cold-import-14 | 3.14 | 源码 | 7 | hotpot.cheminfo.adapters.visualization.draw | 导入/依赖边界 |
| cp314-source-adapter-order-ob-then-rd | 3.14 | 源码 | 1/6 | hotpot；顺序 ob-then-rd | 小样本转换 |
| cp314-source-adapter-order-rd-then-ob | 3.14 | 源码 | 1/6 | hotpot；顺序 rd-then-ob | 小样本转换 |
| cp314-source-missing-ob-conversion-openbabel-before-hotpot | 3.14 | 源码 | 1/6 | ob-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-ob-conversion-openbabel-before-use | 3.14 | 源码 | 1/6 | ob-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-pybel-conversion-openbabel-before-hotpot | 3.14 | 源码 | 1/6 | pybel-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-pybel-conversion-openbabel-before-use | 3.14 | 源码 | 1/6 | pybel-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-rd-conversion-rdkit-before-hotpot | 3.14 | 源码 | 1/6 | rd-conversion；屏蔽 rdkit；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-rd-conversion-rdkit-before-use | 3.14 | 源码 | 1/6 | rd-conversion；屏蔽 rdkit；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-io-read-openbabel-before-hotpot | 3.14 | 源码 | 5/6 | io-read；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-io-read-openbabel-before-use | 3.14 | 源码 | 5/6 | io-read；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-io-write-openbabel-before-hotpot | 3.14 | 源码 | 5/6 | io-write；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-io-write-openbabel-before-use | 3.14 | 源码 | 5/6 | io-write；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-crystal-cell-openbabel-before-hotpot | 3.14 | 源码 | 1/6 | crystal-cell；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-crystal-cell-openbabel-before-use | 3.14 | 源码 | 1/6 | crystal-cell；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-source-missing-thermo-thermo-before-hotpot | 3.14 | 源码 | 7 | thermo；屏蔽 thermo；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-source-missing-thermo-thermo-before-use | 3.14 | 源码 | 7 | thermo；屏蔽 thermo；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-source-missing-thermo-pandas-before-hotpot | 3.14 | 源码 | 7 | thermo；屏蔽 pandas；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-source-missing-thermo-pandas-before-use | 3.14 | 源码 | 7 | thermo；屏蔽 pandas；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-source-missing-draw-svg-cairosvg-before-hotpot | 3.14 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-source-missing-draw-svg-cairosvg-before-use | 3.14 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-source-missing-draw-svg-matplotlib-before-hotpot | 3.14 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-source-missing-draw-svg-matplotlib-before-use | 3.14 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-source-missing-draw-png-cairosvg-before-hotpot | 3.14 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-source-missing-draw-png-cairosvg-before-use | 3.14 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-source-missing-draw-webp-PIL-before-hotpot | 3.14 | 源码 | 7 | draw-webp；屏蔽 PIL；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-source-missing-draw-webp-PIL-before-use | 3.14 | 源码 | 7 | draw-webp；屏蔽 PIL；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-source-prior-cold-unpickle-0 | 3.14 | 源码 | 1/5 | molecule-0.pickle | 持久化/事实恢复 |
| cp314-source-prior-cold-unpickle-1 | 3.14 | 源码 | 1/5 | molecule-1.pickle | 持久化/事实恢复 |
| cp314-source-prior-cold-unpickle-2 | 3.14 | 源码 | 1/5 | molecule-2.pickle | 持久化/事实恢复 |
| cp314-source-prior-cold-unpickle-3 | 3.14 | 源码 | 1/5 | molecule-3.pickle | 持久化/事实恢复 |
| cp314-source-prior-cold-unpickle-4 | 3.14 | 源码 | 1/5 | molecule-4.pickle | 持久化/事实恢复 |
| cp314-source-prior-cold-unpickle-5 | 3.14 | 源码 | 1/5 | molecule-5.pickle | 持久化/事实恢复 |
| cp314-source-p04-prior-cell-cold-unpickle | 3.14 | 源码 | 1/5 | c61-cell-molecule.pickle | 持久化/事实恢复 |
| cp314-source-complete-reflection | 3.14 | 源码 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |
| cp314-source-preserve-surface | 3.14 | 源码 | 1/6（接口） | surface | 接口兼容；非计算流程 |
| cp314-source-preserve-conversions | 3.14 | 源码 | 1/2/6 | conversions | 结构运输小样本；非全规则验证 |
| cp314-source-preserve-io | 3.14 | 源码 | 1/5/6 | io | I/O小样本；非全格式验证 |
| cp314-source-preserve-crystal | 3.14 | 源码 | 1/6 | crystal | 晶胞运输小样本 |
| cp314-source-preserve-thermo | 3.14 | 源码 | 7 | thermo | 替身接口合约 |
| cp314-source-preserve-drawing | 3.14 | 源码 | 7 | drawing | 第三方输出/资源 |
| cp314-source-io-path-string-smi-single | 3.14 | 源码 | 1/5/6 | smi-single.smi；Path/字符串入口 | I/O读回小样本 |
| cp314-source-io-path-string-smi-multi | 3.14 | 源码 | 1/5/6 | smi-multi.smi；Path/字符串入口 | I/O读回小样本 |
| cp314-source-io-path-string-mol2-single | 3.14 | 源码 | 1/5/6 | mol2-single.mol2；Path/字符串入口 | I/O读回小样本 |
| cp314-source-io-path-string-mol2-multi | 3.14 | 源码 | 1/5/6 | mol2-multi.mol2；Path/字符串入口 | I/O读回小样本 |
| cp314-source-io-path-string-sdf-single | 3.14 | 源码 | 1/5/6 | sdf-single.sdf；Path/字符串入口 | I/O读回小样本 |
| cp314-source-io-path-string-sdf-multi | 3.14 | 源码 | 1/5/6 | sdf-multi.sdf；Path/字符串入口 | I/O读回小样本 |
| cp314-source-drawing-independent-cases | 3.14 | 源码 | 7 | 字体、grid、single-string、版本拒绝顺序、10种绘图格式 | 第三方输出/资源 |
| cp314-installed-cold-import-00 | 3.14 | 安装包 | 1/2/5/6（间接） | hotpot | 导入/依赖边界 |
| cp314-installed-cold-import-01 | 3.14 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.structure | 导入/依赖边界 |
| cp314-installed-cold-import-02 | 3.14 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.structure.crystal | 导入/依赖边界 |
| cp314-installed-cold-import-03 | 3.14 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.operations.conversion | 导入/依赖边界 |
| cp314-installed-cold-import-04 | 3.14 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.conversion | 导入/依赖边界 |
| cp314-installed-cold-import-05 | 3.14 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.conversion | 导入/依赖边界 |
| cp314-installed-cold-import-06 | 3.14 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.openbabel.crystal | 导入/依赖边界 |
| cp314-installed-cold-import-07 | 3.14 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.conversion | 导入/依赖边界 |
| cp314-installed-cold-import-08 | 3.14 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.rdkit.stereo_transport | 导入/依赖边界 |
| cp314-installed-cold-import-09 | 3.14 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.operations.io | 导入/依赖边界 |
| cp314-installed-cold-import-10 | 3.14 | 安装包 | 1/2/5/6（间接） | hotpot.cheminfo.adapters.io.formats.openbabel | 导入/依赖边界 |
| cp314-installed-cold-import-11 | 3.14 | 安装包 | 7 | hotpot.cheminfo.operations.thermo | 导入/依赖边界 |
| cp314-installed-cold-import-12 | 3.14 | 安装包 | 7 | hotpot.cheminfo.adapters.thermo.chemical | 导入/依赖边界 |
| cp314-installed-cold-import-13 | 3.14 | 安装包 | 7 | hotpot.cheminfo.operations.visualization | 导入/依赖边界 |
| cp314-installed-cold-import-14 | 3.14 | 安装包 | 7 | hotpot.cheminfo.adapters.visualization.draw | 导入/依赖边界 |
| cp314-installed-adapter-order-ob-then-rd | 3.14 | 安装包 | 1/6 | hotpot；顺序 ob-then-rd | 小样本转换 |
| cp314-installed-adapter-order-rd-then-ob | 3.14 | 安装包 | 1/6 | hotpot；顺序 rd-then-ob | 小样本转换 |
| cp314-installed-missing-ob-conversion-openbabel-before-hotpot | 3.14 | 安装包 | 1/6 | ob-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-ob-conversion-openbabel-before-use | 3.14 | 安装包 | 1/6 | ob-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-pybel-conversion-openbabel-before-hotpot | 3.14 | 安装包 | 1/6 | pybel-conversion；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-pybel-conversion-openbabel-before-use | 3.14 | 安装包 | 1/6 | pybel-conversion；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-rd-conversion-rdkit-before-hotpot | 3.14 | 安装包 | 1/6 | rd-conversion；屏蔽 rdkit；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-rd-conversion-rdkit-before-use | 3.14 | 安装包 | 1/6 | rd-conversion；屏蔽 rdkit；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-io-read-openbabel-before-hotpot | 3.14 | 安装包 | 5/6 | io-read；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-io-read-openbabel-before-use | 3.14 | 安装包 | 5/6 | io-read；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-io-write-openbabel-before-hotpot | 3.14 | 安装包 | 5/6 | io-write；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-io-write-openbabel-before-use | 3.14 | 安装包 | 5/6 | io-write；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-crystal-cell-openbabel-before-hotpot | 3.14 | 安装包 | 1/6 | crystal-cell；屏蔽 openbabel；before-hotpot；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-crystal-cell-openbabel-before-use | 3.14 | 安装包 | 1/6 | crystal-cell；屏蔽 openbabel；before-use；基础夹具 | 故障注入/数据完整性 |
| cp314-installed-missing-thermo-thermo-before-hotpot | 3.14 | 安装包 | 7 | thermo；屏蔽 thermo；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-missing-thermo-thermo-before-use | 3.14 | 安装包 | 7 | thermo；屏蔽 thermo；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-missing-thermo-pandas-before-hotpot | 3.14 | 安装包 | 7 | thermo；屏蔽 pandas；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-missing-thermo-pandas-before-use | 3.14 | 安装包 | 7 | thermo；屏蔽 pandas；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-svg-cairosvg-before-hotpot | 3.14 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-svg-cairosvg-before-use | 3.14 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-svg-matplotlib-before-hotpot | 3.14 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-svg-matplotlib-before-use | 3.14 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-png-cairosvg-before-hotpot | 3.14 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-png-cairosvg-before-use | 3.14 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-webp-PIL-before-hotpot | 3.14 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-hotpot；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-webp-PIL-before-use | 3.14 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-use；基础夹具 | 故障注入/第三方边界 |
| cp314-installed-prior-cold-unpickle-0 | 3.14 | 安装包 | 1/5 | molecule-0.pickle | 持久化/事实恢复 |
| cp314-installed-prior-cold-unpickle-1 | 3.14 | 安装包 | 1/5 | molecule-1.pickle | 持久化/事实恢复 |
| cp314-installed-prior-cold-unpickle-2 | 3.14 | 安装包 | 1/5 | molecule-2.pickle | 持久化/事实恢复 |
| cp314-installed-prior-cold-unpickle-3 | 3.14 | 安装包 | 1/5 | molecule-3.pickle | 持久化/事实恢复 |
| cp314-installed-prior-cold-unpickle-4 | 3.14 | 安装包 | 1/5 | molecule-4.pickle | 持久化/事实恢复 |
| cp314-installed-prior-cold-unpickle-5 | 3.14 | 安装包 | 1/5 | molecule-5.pickle | 持久化/事实恢复 |
| cp314-installed-p04-prior-cell-cold-unpickle | 3.14 | 安装包 | 1/5 | c61-cell-molecule.pickle | 持久化/事实恢复 |
| cp314-installed-complete-reflection | 3.14 | 安装包 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |
| cp314-installed-preserve-surface | 3.14 | 安装包 | 1/6（接口） | surface | 接口兼容；非计算流程 |
| cp314-installed-preserve-conversions | 3.14 | 安装包 | 1/2/6 | conversions | 结构运输小样本；非全规则验证 |
| cp314-installed-preserve-io | 3.14 | 安装包 | 1/5/6 | io | I/O小样本；非全格式验证 |
| cp314-installed-preserve-crystal | 3.14 | 安装包 | 1/6 | crystal | 晶胞运输小样本 |
| cp314-installed-preserve-thermo | 3.14 | 安装包 | 7 | thermo | 替身接口合约 |
| cp314-installed-preserve-drawing | 3.14 | 安装包 | 7 | drawing | 第三方输出/资源 |
| cp314-installed-io-path-string-smi-single | 3.14 | 安装包 | 1/5/6 | smi-single.smi；Path/字符串入口 | I/O读回小样本 |
| cp314-installed-io-path-string-smi-multi | 3.14 | 安装包 | 1/5/6 | smi-multi.smi；Path/字符串入口 | I/O读回小样本 |
| cp314-installed-io-path-string-mol2-single | 3.14 | 安装包 | 1/5/6 | mol2-single.mol2；Path/字符串入口 | I/O读回小样本 |
| cp314-installed-io-path-string-mol2-multi | 3.14 | 安装包 | 1/5/6 | mol2-multi.mol2；Path/字符串入口 | I/O读回小样本 |
| cp314-installed-io-path-string-sdf-single | 3.14 | 安装包 | 1/5/6 | sdf-single.sdf；Path/字符串入口 | I/O读回小样本 |
| cp314-installed-io-path-string-sdf-multi | 3.14 | 安装包 | 1/5/6 | sdf-multi.sdf；Path/字符串入口 | I/O读回小样本 |
| cp314-installed-drawing-independent-cases | 3.14 | 安装包 | 7 | 字体、grid、single-string、版本拒绝顺序、10种绘图格式 | 第三方输出/资源 |
| cp310-source-missing-draw-svg-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.10 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp310-source-missing-draw-svg-cairosvg-before-use-bonded-drawing-supplement1 | 3.10 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp310-source-missing-draw-svg-matplotlib-before-hotpot-bonded-drawing-supplement1 | 3.10 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp310-source-missing-draw-svg-matplotlib-before-use-bonded-drawing-supplement1 | 3.10 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp310-source-missing-draw-png-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.10 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp310-source-missing-draw-png-cairosvg-before-use-bonded-drawing-supplement1 | 3.10 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp310-source-missing-draw-webp-PIL-before-hotpot-bonded-drawing-supplement1 | 3.10 | 源码 | 7 | draw-webp；屏蔽 PIL；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp310-source-missing-draw-webp-PIL-before-use-bonded-drawing-supplement1 | 3.10 | 源码 | 7 | draw-webp；屏蔽 PIL；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-svg-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.10 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-svg-cairosvg-before-use-bonded-drawing-supplement1 | 3.10 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-svg-matplotlib-before-hotpot-bonded-drawing-supplement1 | 3.10 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-svg-matplotlib-before-use-bonded-drawing-supplement1 | 3.10 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-png-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.10 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-png-cairosvg-before-use-bonded-drawing-supplement1 | 3.10 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-webp-PIL-before-hotpot-bonded-drawing-supplement1 | 3.10 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp310-installed-missing-draw-webp-PIL-before-use-bonded-drawing-supplement1 | 3.10 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp311-source-missing-draw-svg-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.11 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp311-source-missing-draw-svg-cairosvg-before-use-bonded-drawing-supplement1 | 3.11 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp311-source-missing-draw-svg-matplotlib-before-hotpot-bonded-drawing-supplement1 | 3.11 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp311-source-missing-draw-svg-matplotlib-before-use-bonded-drawing-supplement1 | 3.11 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp311-source-missing-draw-png-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.11 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp311-source-missing-draw-png-cairosvg-before-use-bonded-drawing-supplement1 | 3.11 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp311-source-missing-draw-webp-PIL-before-hotpot-bonded-drawing-supplement1 | 3.11 | 源码 | 7 | draw-webp；屏蔽 PIL；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp311-source-missing-draw-webp-PIL-before-use-bonded-drawing-supplement1 | 3.11 | 源码 | 7 | draw-webp；屏蔽 PIL；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-svg-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.11 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-svg-cairosvg-before-use-bonded-drawing-supplement1 | 3.11 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-svg-matplotlib-before-hotpot-bonded-drawing-supplement1 | 3.11 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-svg-matplotlib-before-use-bonded-drawing-supplement1 | 3.11 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-png-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.11 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-png-cairosvg-before-use-bonded-drawing-supplement1 | 3.11 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-webp-PIL-before-hotpot-bonded-drawing-supplement1 | 3.11 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp311-installed-missing-draw-webp-PIL-before-use-bonded-drawing-supplement1 | 3.11 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp312-source-missing-draw-svg-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.12 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp312-source-missing-draw-svg-cairosvg-before-use-bonded-drawing-supplement1 | 3.12 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp312-source-missing-draw-svg-matplotlib-before-hotpot-bonded-drawing-supplement1 | 3.12 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp312-source-missing-draw-svg-matplotlib-before-use-bonded-drawing-supplement1 | 3.12 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp312-source-missing-draw-png-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.12 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp312-source-missing-draw-png-cairosvg-before-use-bonded-drawing-supplement1 | 3.12 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp312-source-missing-draw-webp-PIL-before-hotpot-bonded-drawing-supplement1 | 3.12 | 源码 | 7 | draw-webp；屏蔽 PIL；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp312-source-missing-draw-webp-PIL-before-use-bonded-drawing-supplement1 | 3.12 | 源码 | 7 | draw-webp；屏蔽 PIL；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-svg-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.12 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-svg-cairosvg-before-use-bonded-drawing-supplement1 | 3.12 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-svg-matplotlib-before-hotpot-bonded-drawing-supplement1 | 3.12 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-svg-matplotlib-before-use-bonded-drawing-supplement1 | 3.12 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-png-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.12 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-png-cairosvg-before-use-bonded-drawing-supplement1 | 3.12 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-webp-PIL-before-hotpot-bonded-drawing-supplement1 | 3.12 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp312-installed-missing-draw-webp-PIL-before-use-bonded-drawing-supplement1 | 3.12 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp313-source-missing-draw-svg-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.13 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp313-source-missing-draw-svg-cairosvg-before-use-bonded-drawing-supplement1 | 3.13 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp313-source-missing-draw-svg-matplotlib-before-hotpot-bonded-drawing-supplement1 | 3.13 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp313-source-missing-draw-svg-matplotlib-before-use-bonded-drawing-supplement1 | 3.13 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp313-source-missing-draw-png-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.13 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp313-source-missing-draw-png-cairosvg-before-use-bonded-drawing-supplement1 | 3.13 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp313-source-missing-draw-webp-PIL-before-hotpot-bonded-drawing-supplement1 | 3.13 | 源码 | 7 | draw-webp；屏蔽 PIL；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp313-source-missing-draw-webp-PIL-before-use-bonded-drawing-supplement1 | 3.13 | 源码 | 7 | draw-webp；屏蔽 PIL；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-svg-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.13 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-svg-cairosvg-before-use-bonded-drawing-supplement1 | 3.13 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-svg-matplotlib-before-hotpot-bonded-drawing-supplement1 | 3.13 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-svg-matplotlib-before-use-bonded-drawing-supplement1 | 3.13 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-png-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.13 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-png-cairosvg-before-use-bonded-drawing-supplement1 | 3.13 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-webp-PIL-before-hotpot-bonded-drawing-supplement1 | 3.13 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp313-installed-missing-draw-webp-PIL-before-use-bonded-drawing-supplement1 | 3.13 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp314-source-missing-draw-svg-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.14 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp314-source-missing-draw-svg-cairosvg-before-use-bonded-drawing-supplement1 | 3.14 | 源码 | 7 | draw-svg；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp314-source-missing-draw-svg-matplotlib-before-hotpot-bonded-drawing-supplement1 | 3.14 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp314-source-missing-draw-svg-matplotlib-before-use-bonded-drawing-supplement1 | 3.14 | 源码 | 7 | draw-svg；屏蔽 matplotlib；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp314-source-missing-draw-png-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.14 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp314-source-missing-draw-png-cairosvg-before-use-bonded-drawing-supplement1 | 3.14 | 源码 | 7 | draw-png；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp314-source-missing-draw-webp-PIL-before-hotpot-bonded-drawing-supplement1 | 3.14 | 源码 | 7 | draw-webp；屏蔽 PIL；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp314-source-missing-draw-webp-PIL-before-use-bonded-drawing-supplement1 | 3.14 | 源码 | 7 | draw-webp；屏蔽 PIL；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-svg-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.14 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-svg-cairosvg-before-use-bonded-drawing-supplement1 | 3.14 | 安装包 | 7 | draw-svg；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-svg-matplotlib-before-hotpot-bonded-drawing-supplement1 | 3.14 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-svg-matplotlib-before-use-bonded-drawing-supplement1 | 3.14 | 安装包 | 7 | draw-svg；屏蔽 matplotlib；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-png-cairosvg-before-hotpot-bonded-drawing-supplement1 | 3.14 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-png-cairosvg-before-use-bonded-drawing-supplement1 | 3.14 | 安装包 | 7 | draw-png；屏蔽 cairosvg；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-webp-PIL-before-hotpot-bonded-drawing-supplement1 | 3.14 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-hotpot；有键分子补充 | 故障注入/第三方边界 |
| cp314-installed-missing-draw-webp-PIL-before-use-bonded-drawing-supplement1 | 3.14 | 安装包 | 7 | draw-webp；屏蔽 PIL；before-use；有键分子补充 | 故障注入/第三方边界 |
| cp310-installed-core-smoke | 3.10 | 安装包 | 1/2；其余仅资源 | 图环、子结构搜索、甲烷电荷/自旋状态选择、包资源与CLI文档 | 基础语义冒烟；无FF/xTB/MCA推理 |
| cp311-installed-core-smoke | 3.11 | 安装包 | 1/2；其余仅资源 | 图环、子结构搜索、甲烷电荷/自旋状态选择、包资源与CLI文档 | 基础语义冒烟；无FF/xTB/MCA推理 |
| cp312-installed-core-smoke | 3.12 | 安装包 | 1/2；其余仅资源 | 图环、子结构搜索、甲烷电荷/自旋状态选择、包资源与CLI文档 | 基础语义冒烟；无FF/xTB/MCA推理 |
| cp313-installed-core-smoke | 3.13 | 安装包 | 1/2；其余仅资源 | 图环、子结构搜索、甲烷电荷/自旋状态选择、包资源与CLI文档 | 基础语义冒烟；无FF/xTB/MCA推理 |
| cp314-installed-core-smoke | 3.14 | 安装包 | 1/2；其余仅资源 | 图环、子结构搜索、甲烷电荷/自旋状态选择、包资源与CLI文档 | 基础语义冒烟；无FF/xTB/MCA推理 |
| cp310-source-recursive-reflection-supplement | 3.10 | 源码 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |
| cp310-installed-recursive-reflection-supplement | 3.10 | 安装包 | 1/6（接口） | 24个注解/全局条目、28个声明；签名、默认值、类型与身份 | 接口兼容；非计算流程 |

## 证据入口

原始当前执行清单：`/home/zhangzhiyuan/hotpot/.plan20-artifacts/20261103/stage7a/i3-current-native-task234-attempt1/current707-task380-attempt1/activation/functional-release.json`。
探针实现位于同一项目`.plan20-artifacts/20261030/stage7a/validation-preparation/attempt6/`、`attempt7/`和当前绑定指向的缺包补充；安装包core动作对应维护文件`tests/test_packaging/installed_package_smoke.py:18`。
707全部是计划执行清单；当前运行完成数属于单独的时间截面，不在这个静态表中滚动改写。
