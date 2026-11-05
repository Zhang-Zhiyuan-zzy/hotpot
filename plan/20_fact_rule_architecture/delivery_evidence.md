# Plan20 交付证据附表

本附表是[整体报告](delivery_report.md)的按需阅读证据索引，保留62项CPU比较、真实流程、判据、原失败和已接受限制。
业务基线为 `8785a06fca063a179976ca74d6bc8cb14bcce640`；本次发现并修复的 D039/D040 单列在主报告及
[后续修复记录](stages/stage12/final-delivery/validation/postdelivery-interface-cleanup.json)。
历史运行未改称本次新执行，模型数值一致性未改称实验精度。

[机器图](delivery_evidence_graph.json)按 claim → evidence → scope 组织。
完整原始输出仍在各 owning archive；本次汇总只读所需成员，没有重校全部历史归档。

## 证据层级与当前边界

| 证据类别 | 本报告中的用法 | 不能推出的结论 |
|---|---|---|
| 原 `8785a06f` 最终组合运行 | E02、A01 的 incremental/default/五ABI安装、文档/packing/checker | 最终所有源码在同一命令、所有ABI都跑全默认；启用全部科学/外部tier |
| 有效限定复用 | A01 `ROOT_REUSE_DECISION.json`、`ROOT_EXTERNAL_IDENTITY.json`、`reuse-review/REUSE.json`；E02 `native_and_scientific_evidence` | 重建原生/SDK、重跑模型/科学、刷新整个外部库/驱动闭包 |
| 历史原始观察 | E03–E19 与 E46 等原样源快照、方法、样本和失败 | 当前HEAD精确时长、普适随机重复性、所有后续共享owner字节不变 |
| 未执行/未验证 | default137禁用tier、D028可选研究训练/优化/真实CCDC、一般周期多重边领域 | PASS、已有科学正确率或被静默删除的范围 |

A01 指 `.plan20-artifacts/20261105/overall/combined-delivery/evidence.tar.gz`；该维护记录归档有448payload、48101493bytes、SHA256 `a3467c260e12fd8f1379505a72ff03469eae8607c27094ffa711abeddb46c9c8`。本次只读取所需归档成员；没有声称重新校验全部448个payload。

## 原 `8785a06f` 组合基线的源码、ABI和完整流程执行成本

所有下表 wall time 是各次验证控制器成本，不能用于跨解释器性能排序。五ABI受影响57模块执行和独立packing14例是两个显式选择，没有虚构58模块单命令。当前默认之后只加强checker及其测试两个文件；`ROOT_CHECKER_DECISION.json`/`FINAL_SOURCE_REBIND.json`记录业务/构建字节无变，210checker例和仓库扫描新跑通过，复用原 default02 的业务结果。

| 实际运行 | 结果 | 控制器 wall s | 独立范围说明/证据 |
|---|---|---:|---|
| CP311 source 16-module seam |346pass/3strictxfail/0ordinaryskip|17.175325|E02 `incremental_source`；docs 3.066393s|
| CP311完整默认 |8205pass/137tier-skip/3strictxfail；49subtests|399.467928|pytest396.62s；429origins/0violations；E02 `final_cp311_source_default`|
| cp310 installed57模块 |2015pass/3strictxfail/0ordinaryskip|188.653128|222origins/0violations；docs 7.205852s；E02 `final_installed_matrix.cp310`|
| cp311 installed57模块 |2015pass/3strictxfail/0ordinaryskip|133.062849|222origins/0violations；docs 3.230737s；E02 `final_installed_matrix.cp311`|
| cp312 installed57模块 |2015pass/3strictxfail/0ordinaryskip|155.980169|222origins/0violations；docs 7.531505s；E02 `final_installed_matrix.cp312`|
| cp313 installed57模块 |2015pass/3strictxfail/0ordinaryskip|146.182691|222origins/0violations；docs 7.544798s；E02 `final_installed_matrix.cp313`|
| cp314 installed57模块 |2015pass/3strictxfail/0ordinaryskip|147.033548|222origins/0violations；docs 7.786605s；E02 `final_installed_matrix.cp314`|
| 另行packing14例/ABI |14pass×5，70总计|17.611978整项控制器|E02 `validation_corrections.packing.installed_followup`；每ABI168origins/0violations|
| 210-case checker +仓库扫描 |0debt/layer/legacy/duplicate-owner；无allowlist增长|15.018363 checker控制器|E02 `validation_corrections.architecture`；实际1073Python/1111legacy-consumerfile扫描|
| Fresh final sdist |962members/532Python-stub|1.889059|E02 `fresh_sdist`；五wheel各重用6已接受原生payload，属于repack而非新build|

汇总时直接读 A01 的 default02/RESULT、cp311/RESULT 与两个 JUnit，对照与E02一致。默认JUnit testsuite的 `tests=8394` 包含49subtests；实际testcase是8345，其中8205通过、140跳过标记=137tier+3xfail。不能把8394、8345、8205或49相加成不同的化学合同。

本次默认未执行的137tier按原始JUnit精确分组：11需`--run-optional`、44需`--run-model-artifacts`、8需`--run-scientific-boundary`、27需`--run-high-compute`、47需`--run-external`。3个strictxfail原因另计。模型/引擎已有独立真实运行，下文保留其时间身份，不能把本次skip写成已执行。

实际默认命令来自 A01 `default02/tests/intent.json`：

```text
taskset -c 40-47 /tmp/hotpot_plan20/medium/stage9b-pyg-environment-20261104/venv/bin/python -s -B -m pytest -q -ra --import-mode=importlib -p no:cacheprovider -p origin_probe --basetemp /tmp/hotpot_plan20/medium/overall-integration-20261105/default02/pytest-temp --junitxml /tmp/hotpot_plan20/medium/overall-integration-20261105/default02/junit.xml tests
```

cwd为仓库；CPU40–47；OMP/OPENBLAS/MKL线程各1；`CUDA_VISIBLE_DEVICES=''`，禁用自动pytest插件、用户site和bytecode；完整环境在该intent。已安装CP311来自A01 `installed01/cp311/tests/intent.json`，cwd是隔离support目录，使用相同显式pytest选集而非checkout导入；实际argv和每wheel/hash/origin在E02/原归档，未用省略命令冒充完整配方。

原生/科学复用边界经过当前维护记录的按owner审查：37inference、51workflow/backend、40bonding/scientific-rule owner与旧快照匹配；19共享carrier/cell/portable owner不同，靠新14/15及最终seam/default/installed验证。168native-build inputs/setup、46SDK source/header/build/test inputs、19installedSDK和25compiled artifacts匹配。ROOT随后逐文件rehash 88模型payload（MCA85+CBond3）和xTB可执行文件相符；总模型167616128bytes。没有重新验证完整外部runtime-library/driver闭包，也没有推理/科学重跑。19源码工作区二进制与每ABI发布wheel的6原生payload是不同命名空间，不可替换。

## 当前 `2c2ef889` 的 D039/D040 关闭证据

报告核对中的两个具体缺陷已修复：[checkpoint](stages/stage12/final-delivery/validation/postdelivery-interface-cleanup.json)
记录 source308pass/1skip，以及 installed 首次304pass/4missing-support failures/1skip、原样补齐fixture后4pass。
合计308unique passing nodes/1opt-in Gaussian skip；不宣称一次全绿安装运行或此次局部修复的全版本复测。
根代理独立复核源文件、JUnit、wheel delta和236成员归档；所有6个native payload保持原接受版本。
新的8,911,129-byte归档位于 `.plan20-artifacts/20261105/overall/postdelivery-interface-cleanup/evidence.tar.gz`，
SHA256 `94cb3a3ca4b08223787b191394ea12754e0fdf94b3ede9805c8d31b25c1f343a`。
机器图 C23/E51/A02 与原 C22 发现记录分别保留；原模型/科学/性能证据没有变成本次新执行。

## 重构前后性能：保留改善、退化和不可比项

E03 的62CPU配对包含124个phase/workload once-only invocations，双方各771全部尝试、595测量、0失败。每项的原始工作计数、loops、每call值和bootstrap区间可定位 `rows[id=...]`。完整化学可比57项；`clean_read_8/64/256` 返回H信息不同；`mca_cpu_8/64` 虽有相同trained tensor和预测工作量，B0公共酸元数据错误阻止完整输出parity。全部5项仍留在62项与原始分母中。

比率定义是candidate/baseline。下表值全部来自E03，不另造全局GM或speedup。ms/invocation包含该行冻结loops；每call成本另列，避免把1000次空构造误写成一个Molecule72ms。

| E03 rows[id] | B ms/invocation | C ms/invocation | C/B | bootstrap95% CI | C ms/call | 完整化学可比 | 字段起始行 |
|---|---:|---:|---:|---|---:|---|---:|
| `cold_import` |896.947650|555.309813|0.6191|0.6171–0.6198|555.309813|是|109|
| `empty_molecule` |4.100900|72.400155|17.6547|17.3188–17.8561|0.072400|是|157|
| `construct_8` |0.753619|6.634019|8.8029|8.5991–8.9486|1.326804|是|225|
| `scalar_read_8` |0.215630|0.218128|1.0116|0.9928–1.0410|0.010906|是|294|
| `scalar_write_8` |0.066905|0.123083|1.8397|1.8233–1.8562|0.006154|是|363|
| `batch_facts_8` |0.067271|0.178259|2.6499|2.5957–2.7130|0.008913|是|432|
| `coordinates_8` |1.010335|0.415418|0.4112|0.4064–0.4143|0.020771|是|501|
| `insert_delete_8` |0.861073|7.943833|9.2255|9.1212–9.3625|0.397192|是|571|
| `matrix_8` |0.724284|0.158277|0.2185|0.2171–0.2209|0.007914|是|640|
| `native_8` |0.922108|0.396032|0.4295|0.4242–0.4648|0.019802|是|713|
| `partial_projection_8` |0.121300|0.155269|1.2800|1.2528–1.2908|0.007763|是|788|
| `mca_projection_8` |0.040365|0.159211|3.9443|3.8781–3.9744|0.007961|是|857|
| `write_read_8` |1.305101|17.789675|13.6309|13.5339–13.7729|0.889484|是|927|
| `clean_read_8` |0.120449|3.896420|32.3492|31.8835–32.5884|0.194821|否，保留原值|996|
| `unrelated_read_8` |1.714062|0.445545|0.2599|0.2547–0.2632|0.022277|是|1067|
| `construct_64` |11.649626|47.482551|4.0759|4.0194–4.1183|9.496510|是|1136|
| `scalar_read_64` |1.686248|1.691513|1.0031|0.9812–1.0195|0.084576|是|1205|
| `scalar_write_64` |0.505690|0.952402|1.8834|1.8712–1.9153|0.047620|是|1274|
| `batch_facts_64` |0.514276|0.370206|0.7199|0.7085–0.7225|0.018510|是|1343|
| `coordinates_64` |7.254955|0.832193|0.1147|0.1136–0.1153|0.041610|是|1412|
| `insert_delete_64` |2.666770|13.204712|4.9516|4.9105–4.9805|0.660236|是|1482|
| `matrix_64` |5.291457|0.164783|0.0311|0.0307–0.0325|0.008239|是|1551|
| `native_64` |5.033133|0.655619|0.1303|0.1295–0.1308|0.032781|是|1624|
| `partial_projection_64` |0.847192|1.163547|1.3734|1.3523–1.3850|0.058177|是|1699|
| `mca_projection_64` |0.210499|1.050826|4.9921|4.8775–5.0306|0.052541|是|1768|
| `write_read_64` |10.067757|102.450668|10.1761|10.1232–10.2474|5.122533|是|1838|
| `clean_read_64` |0.895103|30.345798|33.9020|33.3870–34.1723|1.517290|否，保留原值|1907|
| `unrelated_read_64` |12.398414|2.928975|0.2362|0.2340–0.2376|0.146449|是|1978|
| `construct_256` |127.981883|187.396431|1.4642|1.4602–1.4724|37.479286|是|2047|
| `scalar_read_256` |6.977886|6.547551|0.9383|0.9226–0.9483|0.327378|是|2116|
| `scalar_write_256` |2.178135|3.866265|1.7750|1.4160–1.8961|0.193313|是|2185|
| `batch_facts_256` |2.006105|1.043134|0.5200|0.5168–0.5255|0.052157|是|2254|
| `coordinates_256` |28.544210|2.171705|0.0761|0.0757–0.0764|0.108585|是|2323|
| `insert_delete_256` |8.712822|30.938677|3.5509|3.5309–3.5857|1.546934|是|2393|
| `matrix_256` |20.126930|0.175563|0.0087|0.0085–0.0091|0.008778|是|2462|
| `native_256` |19.350685|1.546489|0.0799|0.0796–0.0804|0.077324|是|2535|
| `partial_projection_256` |3.426834|4.663629|1.3609|1.3433–1.3709|0.233181|是|2610|
| `mca_projection_256` |0.773642|4.084618|5.2797|5.2420–5.3040|0.204231|是|2679|
| `write_read_256` |40.764309|389.230068|9.5483|9.4753–9.8607|19.461503|是|2749|
| `clean_read_256` |3.528227|120.622779|34.1879|33.7778–34.5682|6.031139|否，保留原值|2818|
| `unrelated_read_256` |49.809491|11.631544|0.2335|0.2325–0.2343|0.581577|是|2889|
| `io_smi_small` |1.288245|5.199619|4.0362|3.8520–4.0844|1.039924|是|2958|
| `io_mol2_small` |1.309730|7.411319|5.6587|5.4542–5.7043|1.482264|是|3028|
| `io_sdf_small` |1.401573|5.360583|3.8247|3.6875–3.8773|1.072117|是|3098|
| `io_smi_typical` |5.520546|9.616371|1.7419|1.7090–1.7787|1.923274|是|3168|
| `io_mol2_typical` |5.571770|17.896407|3.2120|3.1399–3.2500|3.579281|是|3238|
| `io_sdf_typical` |5.870361|10.026474|1.7080|1.6724–1.7168|2.005295|是|3308|
| `io_smi_large` |23.884155|29.802642|1.2478|1.2381–1.2528|5.960528|是|3378|
| `io_mol2_large` |24.992974|65.891706|2.6364|2.6247–3.4889|13.178341|是|3448|
| `io_sdf_large` |25.918463|31.793047|1.2267|1.2162–1.2291|6.358609|是|3518|
| `graph_search_geometry` |10.581102|34.853303|3.2939|3.2744–3.3087|1.742665|是|3588|
| `geometry_screen` |7.389702|6.986889|0.9455|0.9369–0.9510|0.069869|是|3663|
| `crystal_matrix` |2.067978|2.257316|1.0916|1.0844–1.1004|0.011287|是|3731|
| `molecule_bundle` |0.089891|0.089425|0.9948|0.9798–1.0016|0.000894|是|3816|
| `source_conformers` |4.495376|26.164620|5.8203|5.8100–5.8399|1.308231|是|3885|
| `toolkit_conversions` |11.957671|84.695344|7.0829|7.0620–7.1011|4.234767|是|3956|
| `mca_conformer_algorithm` |7.517330|7.439872|0.9897|0.9866–0.9958|7.439872|是|4028|
| `organic_forcefield` |967.960553|780.186748|0.8060|0.8019–0.8133|780.186748|是|4083|
| `retained_molecules` |36.959358|150.589699|4.0745|4.0528–4.0969|150.589699|是|4152|
| `mca_cpu_8` |144.784087|173.894987|1.2011|1.1762–1.4258|173.894987|否，保留原值|4224|
| `mca_cpu_64` |988.028239|1206.198724|1.2208|1.1751–1.2234|1206.198724|否，保留原值|4296|
| `cbond_cpu` |27.975446|27.955459|0.9993|0.9959–1.0068|1.397773|是|4368|

这张表给出真实的权衡：冷import/坐标/矩阵与native export/部分批量读取改善；空构造、标量写、插删、dirty-H读取、MCA结果投影、toolkit conversion等显著变慢。E03原决策按绝对成本和实际任务可完成性接受可用成本，没有“约5×以内必过”的数值豁免。所有62worker RSS峰低于B0，但39项单独Python-tracked peak更高；不是完整进程树、GPU或泄漏证明。

| E04 GPU/fallback完整实际工作 | B median s | C median s | C/B | 必须保留的条件 |
|---|---:|---:|---:|---|
| `mca_fallback`：8molecules，未启动GPU |2.414037989|2.605925923|1.079488|每项1warmup+3measurement，0failed/no-output；保留adverse原值|
| `mca_1gpu`：512molecules/768sites/2176atom predictions，1device |2.611551125|4.844943319|1.855198|每项1warmup+3measurement，0failed/no-output；保留adverse原值|
| `mca_4gpu`：512molecules，4initialized sessions，仅2device实际推理 |2.317406153|4.653947297|2.008257|每项1warmup+3measurement，0failed/no-output；保留adverse原值|
| `cbond_cuda`：20实际CBond calls |0.055648657|0.056297248|1.011655|每项1warmup+3measurement，0failed/no-output；保留adverse原值|

GPU成本也绑定B0/16b，非精确ED7/最终HEAD时长。共享同一pinned Docker/device上下文，36硬件观察中load1最高21.8；GPU0/3有其他约13GB常驻，最少空闲各33185/48137/46347/35197MiB，clocks/load变化；粗粒度采样不是设备预约。B0有129个酸元数据差异，不能靠work counters说完整化学输出相同。4GPU功能复测确有4device计算（下文E13），与这个成本fixture只有2active须区分。

| 测试成本及新机制 | 实际前后/当前值 | 结论与限制 |
|---|---|---|
| 冻结共同测试collection |B9.065757→C11.455639s；raw1.263616×|E05 `cells`,`raw_comparison`；3948logical→3945physical、3retired，合同/skip不同，0strict comparable pairs|
| 冻结共同测试execution |B238.021692→C230.868108s；raw0.969946×|4cells各1warmup+5测量，全24尝试完成；按phase连续非交错，无strict speedup/parity资格|
| 新增3183选集 |每次3169pass/14tier-skip；median121.524642s，MAD1.021118，p95122.502134，RSS722071552bytes|E07 `cost`；1warmup+5测量；current-only16b，含probe/origin/path优先级开销，不能加到其他时长造总套件成本|
| 新marking/full19投影 |12cells/36warmups/120测量/12memoryprobes；N4096 full19=72.191ms/read|E08 `usable_costs`；无B0对照；full19不同于冻结selected-field projection|
| 完整256C两frame SDFwrite |3warmups+10测量；20calls/batch median0.646559s→32.327960ms/call|E09 `plain_cost`；581publicwrites/1162record全部持久化并独立读回，current-only|
| 独立320write/640frame资源窗口 |Pythontracked+48321B，RSS+1081344B，peak152969216B|E09 `root_dispositions.P20-D004`；17checkpoints/16batches；含输出/observer bookkeeping，不归因为特定leak；历史3264B step/native owner未明|

版本成本E06原2689logical选集共60尝试（5ABI×2phase×(1warmup+5samples)）：baseline30成功、candidate原30旧reference checker失败。下表失败时间只能记失败尝试成本，绝不能输出“各ABI成功加速比”。纠正后的CBond真实case另有各ABI直接pass，未为旧checker重跑通过cohort。

| ABI | B 成功median s | C 原失败checker尝试median s |
|---|---:|---:|
| 3.10 |199.190778|146.771161|
| 3.11 |95.889216|93.498795|
| 3.12 |207.116663|145.843238|
| 3.13 |216.379303|144.958542|
| 3.14 |230.096782|157.021243|

E05硬件监测在baseline执行后期才启动：candidate和两次collection没有timestamp-matched样本。覆盖的baseline12点load1=1.72–5.83、CPU频率1916272–4209016kHz；这不是无争用全局比较。E06按CPU0–39/每ABI8core共享运行；跨ABI时长不可直接解释成解释器优劣。

Stage13的七cell原配对结果（E19，154顺序AB/BA timed phases、28freshmemoryprobes，0失败/重试）：

| cell | C/Bmedian | pairedbootstrap95%CI |
|---|---:|---|
|rule.benzene|1.0283|0.9886–1.0653|
|rule.fused|0.9273|0.8258–1.1312|
|rule.coordination|0.8776|0.7416–1.0260|
|rule.non_aromatic|1.2021|0.7502–1.6501|
|rule.unresolved|0.9873|0.9744–1.2305|
|workflow.explicit|0.9726|0.9610–1.0304|
|workflow.lazy|0.9897|0.9712–0.9951|

benzene+2.83%、nonaromatic+20.21%不利median保留，区间跨1方向不确定；lazy小五pair局部区间低于1不变成全局加速。原20freshmolecule explicit/lazy117.291/189.620ms可用，但后续carrier变化使其仅是原实验工作流时长。CPU64–71频率1717.443–4186.279MHz；nonaromatic200call净retained+754B，lazy20molecule+593B；单probe无不确定性估计，RSS增量0不是无分配/常数空间。D037保持accepted limitation。

## 科学判据：规则、阈值、独立参照、失败控制

以下阈值是当前指定方法与声明域内的判据，不能用于判定所有化学结构是否存在。`AcceptanceCheck.severity='error'`意味着该质量报告失败；不自动等同不可继续的hard物理定律。finite、映射/归属、状态合法性等hard完整性约束局部拒绝；非hard几何/收敛发现默认保留有限观察，用户可按精确代码选择sample/operation break。

| 业务判据 | 实际规则/阈值 | 独立正、反、边界参照与实际局限 | 最小证据位置 |
|---|---|---|---|
| 边界身份与atom/site映射 |facts identity含membership/isotope/H/semantic graph；geometry另digest；结果须绑定同一identity，reorder需显式map|坐标变化只失效geometry；加H/离子改变facts；MCA比较完整ordered rows、元素、source-order与site→atom值join；canonical SMILES重解析不恢复原row|E37全文；E13 `independent_oracles`；E16 `actual_models.business_oracles`|
| 净电荷 |完整声明时 Q=Σatom q=Σcomponent Q；仅complete evidence enforce；explicit0不同unknown；冲突CONFLICT、缺失UNRESOLVED|正/负/显式0及partial assignments；neutral acetic acid charge0/H4、opposite lactic acid charge0/H6；不为继续而猜neutral|E38；E16 `actual_models`；E42的knownH0与positiveH/applicability cases|
| H物理计数 |T=site target，E=graphH，R=max(T−E,0)；toolkit新输入T=E+R|methane E0/E1/E4→T4/R4,3,0；48lifecycle states手数inventory/field14/CBond9MCA encoder+9collated；MCA只接受4state，44graphHstate明确拒绝；不把拒绝tensor探针算模型推理|E18 lines首段/featureprobe；14normalization需定制checkpoint重生成输入/重验|
| 不完整Lewis H域 |固定bond主族provider在non-H site totalH=None时`PROVIDER-001.INCOMPLETE_HYDROGEN_INVENTORY`，NOT_EVALUATED/OUT_OF_DOMAIN，无candidate|8actual-input controls；knownH0、positiveH、explicitH与unknown区分，H2自己的None允许；不宣称原算术错误|E42；所属D021已关闭|
| xTB电子/UHF合法性 |Ne=ΣZ−Q，0≤u≤Ne、(Ne−u)%2=0；若纯spin最高MS约定则u=2S=M−1；ECP另metadata|evenNe/u1、oddNe/u0、negativeNe/u>Ne在backend probe/launch前拒绝；合法显式u可与unresolved scientific spin并存；选中spin冲突不能靠DEFAULT004覆盖|E39全文；E40控制合同；GFN-FF只charge，无伪造spin；GFN2 UHF0执行默认不等于基态|
| 坐标与数值完整性 |shape=(atom_count,3)、所有coordinate finite；final_energy、final阶段rms/maxgradient须finite；未知单位/映射/归属不可用|NaN/Inf保留named cause/artifact，不能返回伪成功；未知程序异常向上传递；数值有限仍可几何不佳|E32 `_coordinate_acceptance_checks`、`_forcefield_acceptance_checks`；E40|
| 标准几何接近/键长监控 |overlap1e-3Å；basic所有pair<0.40Å；standard非键pair<max(0.50Å,0.55×radius_sum)；0<显式键长≤30Å；共价ratio0.65–1.45、金属配体0.65–1.60|真实case21/root44 C28–Am54=1.01906899966Å<1.408Å；旋转平移不变、测量abs1e-12验证；默认finite返回，与显式`FF:atom_too_close`samplebreak都保留3inputs/3attempts和trailing成功|E31 `StructureAcceptanceThresholds` L86起；E32 L327/359/480；E34 L39–125|
| 收敛/strict额外条件 |standard final convergence为warning；strict为error；strict rms≤1、max≤5 kJ/(mol·Å)，ΔE≤1e-4 kJ/mol、maxdisplacement≤1e-4Å，window5|本科学corpus用standard，不能把strict门槛追认为该run实际判据；未收敛finite值仍留；没有跨不同方法能量阈值|E31 L95–100；E32 L83–167|
| 磷fixture意图 |neutralP、4bonds、bond-order sum5、恰1个P=O/S；每对邻键angle sine finite且>1e-3|15inputs×3seed×source/installed；此预有不塌缩monitor不是四面体理想角/唯一结构oracle；原一次UFF极端能量只证明transport|E44 `geometry` L186–211；E17 phosphorus段；没有新能量合格线|
| bond-ring surface |confirmed piercing使质量报告fail；undetermined和超scope分别warning，不能把未测当无piercing|正/反/边界fixture覆盖confirmed/undetermined/excluded/complete scope；容差abs length1e-8、rel1e-10、parameter1e-10、winding residual1e-10；surface枚举budgets保留不确定性|E33 L23–35/L81–87；E50 L260/285/307/331；E32 `_bond_ring_acceptance_checks`|
| MCA数值repeat/域 |repeat rtol1e-5、atol0.001kJ/mol；neutral heavyatom 2–512、site rows1–4096；charged需显式out-of-domain optin；graphH不是预测目标|同predictor重复实际maxdiff0仅本scope；Opposite stereoisomer不强制相同分值；模型FP16release与PyTorch parity另列，不是实验误差|E13 `repeat_tolerance`；E26；E16 `not_required`；E30 charged/applicability|
| CBond模型/化学参照 |第一步score>−0.5，后续>−0.125；rawScore为logit；Prob是terminal donor-index states路径sigmoid乘积汇总归一化|aminodiol原C3H9NO2/H9/Q0、5原logits、2weights、原5e-6容差与donors保留；连接numericSINGLE，不伪改DATIVE；未指定stereo不选异构体；greedy可不是Rank1|E29 thresholds/path定义；E15 `chemical_contract`,`retained_failures`|
| 立体空间证据 |原case54的两中心声明R,R必须在独立保存坐标oracle中保持；不只metadata pass|原错误S,S保留；ED7 reference1+rebuiltUFF2+actualxTB6geometry均独立R,R；complex两xTB路线仍P–S ratio2.596239/2.393746>1.45，finite但failed_final_quality|E14 `actual_affected_science.coordinate_oracle/actual_routes`|
| Crystal代数/metric/maps |Fraction精确affine/fullgroup闭包；metric rtol/atol1e-10Å²；wrapping仅1e-12整数邻域snap；坐标重构1e-10Å，exact maps/discrete equality|230groups/4425ops/284089pairproducts，加独立screw/glide/offorigin/large-rational；skew正确image(−5,0,0)超27stencil；完整一个placement bijection，无free fit/逐atom wrapping|E23；E35 numericalpolicy；E36 metricbounds；E47 independentfixtures|
| finite closed-loop linking |实际directed bonds+image闭合、voltage0、分离simplefinite curves；有界128bitdyadic角区间仅包1整数才返回integer|analytic±Hopf/plane-separatedzero/Whitehead非平凡0/反向交换细分刚体/极端scale/边界交触；None+named unavailable保留原因，0非unlink证书|E24；E36 finite linking；引用Bertolazzi/Ghiloni/Specogna arXiv1912.13121v1与DeWit/Kauffman/Links math9811128v2p17为维护记录，未本次外网核验|
| CIF/PyG 当前事实 |CIF标准与hotpot.cif/1 extension需语义一致，1e-10Å/metric容差；ordinary Data schema2 edited tensor authority，unknown≠0/P1|foreign discrete6.5不能截6；occupancy/isotope/H/stereo/reordered join、standalone vs empty、present molecular declarations独立边界；不启动隐式chemistry/model、不反序列化任意对象|E20 `review_and_repairs`叙述、E21；final seam E01/E02|

hard/local与nonhard/control的实际完整例：E43真实mixed FF/xTB各3inputs→2outputs，保留中间FF-coordinate request与xTB UHF parity cause、后一个样本成功；E34三样本geometry默认返回3，显式samplebreak返回2且保留中间SDF观察；它的optimizer被patch，只证明真实geometry发现后的控制，不是新一次优化。E41 legal-state retry默认disabled、opt-in有独立budget/attempt workspace，不由错误字符串造charge/spin，不自动把后来收敛当基态。

## 真实科学执行与模型依据

| 方法/种群 | 实际观察 | 科学解释与原始证据字段 |
|---|---|---|
| Plan12 inactive corpus |2280objects/replay；inactive profile|E45科学表；仅inactive，不伪称自动profile已激活。集成记录是需求inventory，不是执行证据；最终accepted组合记录统管现状态|
| Eu参考 |187inputs→182natural CBondeligible；181monitorpass/1qualityfail(case61)/5CBondreject(139,182,185,186,187)|E10 `observations`；wall116.877818s，16workers，100×100预算；873733submittedsteps/967initialization/8747epochs/12336frames；37convergencewarnings、1269bondring warnings；不是正式性能样本|
| 独立ligand+complex UFF |369dynamic targets；6qualityfailure；17359frames/3406314coordinate scalars全部finite，selected/finalenergyfinite|E11 `observations`；187ligand+182complex含case61，不用质量筛掉目标；wall159.773296s；ligand176converged/11nonconverged，complex148/34；实际rebuild所有182topologies，SHA只证明transport|
| 直接GFN2与GFNFF→GFN2 |738eligible routes/1105actualoptimizerlaunches；1088finiteenergystages；691monitorpass/30failed_final_quality/17backendfail，738都保留final或lastfinite structures|E12 `route_outcomes`,`stage_outcomes`,`retained_structure_counts`；xTB6.7.1/edcfbbe，executableSHA271dd6…；两方法最多Z86，5complex无CBond→10noteligible routes；共享instrumentation不是正式性能|
| O0/Am三seed版本观察 |187×3=561slots，546complete/15ineligible/0executionerror；545qualitypass/1unchangedcontactfail；107convergencewarning/24undetermined/9incomplete scans|E17；519selected pairs exact、27pairs/13cases downstream变；552providerbuildcomplete arrays/physicalgraphs exact；originalhash comparator561exclusions仍保留，独立实际fields重比另有范围|
| 磷publicUFF |15inputs×seeds42/43/44=45observations每source/installed；各84selected/84terminalPsites、45qualitypass、43selected/45terminalconverged|E17/E44；选中与terminal角色修复，2selectedwarnings仍在；能量412.592–1086.203kJ/mol；OBBuilder忽略seed，所以3标签不是独立RNG实验；exactnativecompletedsteps不可得|
| finalMCA真实CPU/GPU/routing/rolling |12cases/19calls/8959独立化学rows、0observedchemicalproblem；129085events审阅|E13；255sites走CPU、256启动CUDA；warmmiddle smallcall留sessions但走CPU；repeated4GPUcase4device都有CUDA，gpu_count_4单次仅2active；不把性能fixture混入|
| ED7受影响MCA/backend |实际MCA4pass；7composedbusinesspass；case54reference/UFF/sixxTB坐标R,R|E14；原business6pass/1缺README-support失败，精确1node重验pass而非原command转绿；有限complex质量失败保留|
| 原CBond化学reference/CLI |allfiveABI各1实际correctedcase通过，current两CBondfile+READMECLI/FF36pass，0skip|E15；原30matrix失败+candidate1错误no-explicitH/candidate2错误DATIVE假设保留；没有改原numericlogit/tolerance/model/feature|

实际xTB路线拆分，不把后端失败和几何monitor失败混合：

| target / route | monitor passed | failed_final_quality | failed_backend |
|---|---:|---:|---:|
| ligand/direct-gfn2 |180|4|3|
| ligand/gfnff-gfn2 |181|4|2|
| complex/direct-gfn2 |161|13|8|
| complex/gfnff-gfn2 |169|9|4|

参考Eu与旧记录有5个finite终能变化仍因果未知：case42/69/70/71/83分别−8.7889243772/−31.7196420124/+0.0000132459/+3.5166909629/−0.0679416999 kJ/mol（E10 `limits`）。不能用固定历史相等当成功条件，也不能把差异宣称改进。O0另有两个比较：历史B20→Stage8B33个差异；当前Stage8B→Stage12C27个差异，后者|ΔE|median/p95/max3.142/534.503/1103.404kJ/mol，same-rowproperrotationRMSD0.031633/0.597302/2.456609Å（E17），模型bytes相同但wrapper不同、native/electronic state未完整归因。这些比较不可合并计数或归因为RNG。

| 模型/方法参考身份 | 已维护实证 | 应避免的扩大解释 |
|---|---|---|
| MCA v1 MeCAP，Uni-Mol v2 84m单atomregression，FP16，kJ/mol；pinned HF revision `adf525b6a95d6d296ca0d5c68995b1a0ce14d5fe` |E26/E27；ONNX SHA39f01a6c…，83externalweights；62strict-neutral Mayr comparison/storedconformers对PyTorch：FP32RMSE0.000049/max0.000153，FP16RMSE0.087672/max0.233032kJ/mol；INT8max约657.73被拒|这是release数值parity历史记录，本次未重跑，不是MayrN/sNN误差/实验准确率；模型主要neutral域，charged不适用估计须显式选择。未提供新的独立化学heldout/误差校准验证|
| CBond v1 graph+dynamichead，opset21；同pinnedrevision |E28；graphSHA55a3e77c…/headSHA871b5881…；28fixedshape权重等同输出max2.384185791015625e-6/RMSE6.881243244406242e-7；10bundledgraphs/222candidatebonds endtoend max3.814697265625e-6/RMSE9.69567347832232e-7|这是模型导出一致性，不能说coordination预测实验准确率；最多32rings/64atoms perring；单metal+ligand连通性，不是bindingenergy/geometry/thermodynamicstability|
| MCA/CBond本次化学join依据 |E16中性酸和两个对映异构体独立parsedH/Q/orderedatoms/fullsitejoins；E13全部8959rows独立化学检查；E15 aminodiol独立unsanitizedliteral参照|不制造MCA数值golden；不强制不同对映异构体相同预测；所有917历史酸metadata失败保留，新pass不重写历史|

## Crystal/PyG 当前领域、完整工作成本与外推限制

| 工作/判据 | 观测 | 源与范围 |
|---|---|---|
| SpaceGroup/SymmUnit |230referencegroups/4425representatives、284089pairproducts/fullcarries/inverse；independentscrew/glide/offorigin/large-rationalfixtures|E23；reference穷举是代数/表证据，不是独立每entry科学oracle；P1/512positions2.893ms、P21/1024 2.865ms、Fm-3m/98304 6.226ms、F225primitive roundtrip68.330ms均原checkpointcurrent-only|
| 完整cell/finiteunits/supercell |group19，8source sites23.212386/123.693887/225.927480ms；32sites51.837108/216.945084/843.991783ms|E46；1warmup+5calls+memory，1536atoms/1488bonds expansion；后续helpers变化，只保留原generation成本，不写最终union时长|
| 完整placedrelations |32/16sites candidates17.140570ms，128/64=31.450780ms；symmetry16sites44.737490ms、64sites64.688249ms|E47；一个completeplacedbijection，同位素/charge/occupancy/disorder/dative/stereo相容；不fit、不per-atomlift，当前原方法scope|
| finite-loop publicquery |4/8/16vertices每loop13.089115/49.048528/189.000507ms|E24；精确all-pairquadratic，sharedload9.25/3.12/1.57；无large-loopthroughput保证|
| periodicgeometryquery |inversion2/1=4.305437ms；boundary6/5=8.179175ms；Hopf8/9=23.445208ms；sparse64/256/1024sites=93.999794/388.714593/1861.170750ms|E22；完整capture/preparation/matching/surface/link work；denseoutput可quadratic、skewlargecutoff多images；maxtraced7.616529MB仅Pythontrace|
| CIF真实corpus |11source observations全保留：10supportedread/write/read、AFEJOK_ion_b1局部fullprojectionreject；107sites/138bonds、25repeatedunorderedendpointpairs|E21；symmetry2有14nonzero codes与dot，旧all-dot简称错误保留。一般self/parallelquotient D038deferred；不collapse、不扩胞、不合成Atoms，source保留|
| CIF完整468site AFITUF |read18.892934ms/write17.999179ms/read-write-read80.304334ms；完整tracedpeak3068035B|E21；cell-onlyroundtrip40.227761ms、16siteP21 51.273121ms；1warmup+5calls+memory，非旧六scalar比较|
| ordinaryData完整export→Batch→CPU/unbatch→freshimport |mixed16/32sites60.301725ms；crystal4×128=66.888210ms；generated128=201.867601ms|E20；tracedpeak2251280/3064387/2764866B；RSS373372/378916/379180KiB是所有fixtures预建同process累计值|
| generalCUDAtransport |A800GPU6，5graphs3/16/0/4/0sites，32syntheticSGDsteps，重构/trustedlocal保存加载通过；172origins/919unchangedproductfiles|E20；PyG2.6.1/Torch2.7.1+cu126/CUDA12.6/Python3.11.14；Molecular更早3graphs4/128/0sites、161origins另scope；不是deployedmodel科学或performance|

Stage14/15 finalgeneralPyG是schema2，旧schema1与`to_pyg_data`没有loader/alias；editedtensors是当前事实；泛型observations导入后保持historical，不能自动恢复selectedMCA/partialcharge。CIF不承载全部conformer/MCA/runtimeextras，实际omissions emit precise findings。Crystal exact/group/currentimagefacts与historicalmaps有别；unknowngroup不变P1，missingimage不变0，standalone不等于附着emptyMolecule。Molecule仍是simplegraph/Cartesian唯一owner；无禁止的OB/RDKit/CCDC Python Crystal路线。

Stage14成本在共享3995WX64core/128logicalCPU host、affinity0–127、OMP/BLAS/Torch各1thread，load0.32/1.59/1.42→0.47/1.56/1.41；tracemalloc同process单独phase遗漏大量nativeTorchallocation、RSS累计。CIF/geometry各自保存hardware/load；没有专用机器/漏内存自由/新组合globalbenchmark结论。

## 必须保留的原失败、已知问题与下一计划可执行改进

| 已知问题/原失败 | 当前状态/实际影响 | 下一计划建议与可验证完成条件（建议，不改变当前接受范围） |
|---|---|---|
| D001/002/003/004/005/017/018/019/025性能资源 |有界成本acceptedlimitation；storage/H/graph/view/stereo/I/O/constraints/search/state都有不利记录；D005 axis6.276–19.854×、bondexport2.651–2.741×；D025233adverse观察含concurrentcompile1.252×保留|按实际用户代表batch识别热点，先profile empty/insert/Hdirty/conversion/resultprojection与完整workflow；冻结工作/方法/资源再成对采样，保留不利样本，拒绝减少化学工作制造加速；有OOM/不完成/增长即reopen|
| D022未解释数值变化 |33历史与27当前差异分开，107nonconvergencewarning/coveragegaps保留；未证明RNGcause|后续改动或扩大科学声明时，用固定同input/state/backend/context的受控重复保留每attempt、native/electronicstate/seed生效情况，判断稳定性/分布；不能追求固定历史successcount|
| 3 strict芳香限制 |cycle-orderdependence、missingglobalfusedazulene、sharedpathjointringgrouping；finalsource及5installed都保留strictxfail|单独化学算法项目，使用独立positive/negative/boundary与等价表征、fused/桥连/配位cases；明确允许变化与既有stereo/H事实边界；不要借Stage13privatebinding变另一个algorithm|
| nativeSP→RDKit等限制 |registered不支持族/组明确reject；rawnativeLevel2不等于CIP；已修复TH/CT/directbuilder物理stereo与materializedH |在需要SP能力时单独定义transport语义、handedness/placement与独立coordinateoracles，做实际model/FF/installedABI受影响验证，不以metadataonly pass代替几何|
| D034wholemolecule admission |rolling4096admission8195vsnominal8192，resident10241vs32768，pending2/8，全部creditsdrained；名义值是pre-admission阈值|若要strictpostadmissioncap或largermolecule/memoryclaim，先定义wholemolecule策略再测输出顺序、batchshape、peakprocesstree/GPUbytes、完成与OOM，不能把阈值当当前hardcap|
| D028optionalresearch/第三方 |可选training/optimization/realCCDC未验证；D012thermo/dataset/correlation/fitnesssentinel/vis等独立限制，不含mandatory依赖豁免|在任何新capability声明或接mandatorypath前，补真实依赖/engine的source与installed成功/失败归属与scientific观察；finitefailuresentinel不可当measuredfitness|
| D036privateASTchecker |三轮失败于历史count/wildcardscope/tuplelist export comparison；无证明productdefect；没有all-memberextractorPASS|明确“不做第四次retry”；后续集中toolcleanup若确有必要再处理，沿owner/actualbusiness/APIcontracts形成有限可维护audit；不新造checkerapprovalgate|
| D038一般周期self/parallelquotient |DEFERRED；当前simpleimage-labelledgraph以外fullprojection preciseCRYSTAL-UNSUPPORTED-PROJECTION、source/denominator保留|若真实network需求成立，另界定cross-stage soleindex/writer/native/model影响与原子/边image身份；独立self/parallel/periodicpositive/negative/boundary及CIF/PyG/native handoff验证，无edgecollapse/自动扩胞/secondchemistrygraph|
| finaldefault01 |8191pass/2fail/137skip/3xfail；oneULPprojectedanglestrictequality与30architecturefalsepositives是validationmismatch |已纠正packingrtol0/atol1e-12并保留basis/metric独立checks；checker精确reviewedreceiver/value/facade scope；原失败保留，不声称businesssciencerepair|
| Stage14costattempt01 |disposableMoleculeRecord(eq=False)identity误比较；首case完成1warmup/5timing/1memory但raw未flush不可得；无productmutation|原失败与missingraw明确；已用24fieldsvalue/ownership比较重做attempt02；今后仅在真实测量流程中先flushraw再assert，不能补造原sample|
| finalsdist02postcheck |backend实际buildpass，私有postcheck错要求manifest外3tests；962shippedpayload同accepted01|保留原failedpostcheck；无业务/包装修改、无额外重试；只说包payload一致，不把tarcontainerhash相同当事实|
| MCA历史917酸metadata失败/D032/33/35已修复 |旧失败不抹；新source/installedrealmodel、directnativecoordinate与5ABIaffectedqualification支撑修复|将中性酸/手性/显式H物理inventory、rawprediction与atomsitejoin作为受影响回归；不造MCAgolden或跨stereoisomer equality|
| defaulttiers/新科学精度 |137本次skip未执行；模型card与导出parity是历史数值一致性；generalSGDtransport不构成科学训练验证|未来若声明预测科学精度/校准域，预声明独立heldoutcorpus、methods/states/units/tolerances、per-siteworstcases/siteclassification变化、coverage/OOD/失败，actualsource/installedmodel执行后发表，不用工程pass数替代|
| 报告核对发现的旧MCA指南措辞 |原README有宽泛TH/CT转换限制及“high-confidence mapping”措辞；当前支持证据为E14/E16 |本轮已对齐到注册OB TH/CT/SP、RDKit TH/CT及明确SP/未登记拒绝域，site-filter不再暗示校准confidence；模型/算法未变 |

E17原recorder因Atom._attrs删除在优化前45slots失败，是观测工具错误，不能记成45次化学优化失败。Stage14最初4setup因fixture未知keyword失败、0timedcalls保留；实际CUDA前2次缺rich/Cython import失败，第三次依赖齐全执行成功。E19最初focused7pass/2fixturefail后9pass，连接production固定。更多原错误/业务修复与范围见E16/E20/E21各拥有记录，不能把修复前失败重命名为通过。

## 全部38项原有问题的完整状态盘点与报告后续修复

以下计数直接逐行解析E25的D001–D038，不把历史“仍open”叙述误作现状态：**18 CLOSED、19 ACCEPTED_LIMITATION、1 DEFERRED，共38**。严重级别是**S0=2、S1=9、S2=18、S3=9**。交叉分布如下：

| 严重级别 | CLOSED | ACCEPTED_LIMITATION | DEFERRED | 合计 |
|---|---:|---:|---:|---:|
|S0|2|0|0|2|
|S1|9|0|0|9|
|S2|5|12|1|18|
|S3|2|7|0|9|

这38项里没有OPEN/IN_PROGRESS/RESOLVED；19个acceptedlimitation不是19个通过，D038也不是被删除范围。全部当前条目清单如下（业务影响与证据仍以E25该ID整行为准）：

| ID | 严重级别/分类 | 当前状态 | 业务意义/可复核边界 |
|---|---|---|---|
|P20-D001|S2 / performance|ACCEPTED_LIMITATION|代表storage/H/graph/view成本及高trackedretention接受；full19新机制另计，无parity/scaling/leak承诺|
|P20-D002|S2 / performance|ACCEPTED_LIMITATION|72原stereo不利metric与chain512不确定性保留；实际修复R,R/O0/Eu/UFF/xTB限定完成|
|P20-D003|S2 / performance|ACCEPTED_LIMITATION|581writes/1162records、32.328ms完整两framewrite可用；无新B0成对比较|
|P20-D004|S2 / resource attribution|ACCEPTED_LIMITATION|320write资源窗口tracked+48321B/RSS+1081344B；3264B历史step/native归因未知|
|P20-D005|S2 / performance|ACCEPTED_LIMITATION|constraints/topology成本退化与非线性采样保留；代表FF/xTB可完成|
|P20-D006|S1 / version adaptation|CLOSED|Python3.10–3.14声明source/installed/native边界组合完成；skip/xfail/旧失败保留|
|P20-D007|S1 / native delivery|CLOSED|ED7同sdist五ABI各6原生cleanbuild+release/linkage+受影响安装；SDK按scope复用|
|P20-D008|S1 / package contents|CLOSED|41原package遗漏逐项归位；22source/7installed与真实资源消费者，非可选workflow验证|
|P20-D009|S3 / optional backend optimization|ACCEPTED_LIMITATION|privateGTD优化未ship；仍官方installedOB3.2.1，无privatepatch加速声明|
|P20-D010|S3 / simplification|ACCEPTED_LIMITATION|9defense候选A2/A7退休，其余7保留窄duplicatecheck；无已证核心缺陷/瓶颈|
|P20-D011|S2 / core-adjacent semantic assessment|CLOSED|xTB严格JSON/finite、frame/Gaussianhistory、null精确cause等按实际installed边界关闭|
|P20-D012|S3 / independent integrations|ACCEPTED_LIMITATION|独立parser/thermo/dataset/correlation/fitnesssentinel/vis限制；无可选科学能力背书|
|P20-D013|S2 / diagnosed cache symptom|CLOSED|cache仅已命名permission/diskerrno可临时copy且警告原因；unknown/integrity异常传播|
|P20-D014|S3 / validation tooling|ACCEPTED_LIMITATION|旧preterminalreceipt/Gitindex完整性/comparatorpolicy不再验收authority；失败与不可比H原值保留|
|P20-D015|S3 / record-layout tooling|ACCEPTED_LIMITATION|layoutaudit448assets仍1个historicalbareSMILES误报/exit1；36focusedpass，无clean扫描pass|
|P20-D016|S3 / independent Gaussian producer|CLOSED|Gaussian实际parser/update发布完整观察/selectedMulliken、rollback；真实Gaussianengine未执行|
|P20-D017|S2 / performance|ACCEPTED_LIMITATION|frame/conformer/rollback成本接受，N64两frame1.308ms/5.820×；原大frame与allocation保留|
|P20-D018|S2 / performance|ACCEPTED_LIMITATION|canonicalstereo/rollback采样N256scalar4.512ms/155621B、full4.289ms；无普适parity|
|P20-D019|S2 / performance|ACCEPTED_LIMITATION|scientificstate完整copy/Hprepare保留；extract0.782–1.070/resolve0.337–0.338/project0.049–0.051ms|
|P20-D020|S1 / scientific snapshot semantics|CLOSED|保留presentCURRENT H0为0，unknown/absent和conflicterror不混淆|
|P20-D021|S1 / provider applicability|CLOSED|Lewis只在complete-H已声明域内enumerate；unknown非Hsite-H namedabstention|
|P20-D022|S2 / scientific reproducibility assessment|ACCEPTED_LIMITATION|历史33与当前27数值变化原因未知；545quality/1contactfail、107warning/coveragegap保留|
|P20-D023|S2 / phosphorus validation coverage|CLOSED|15P×3seeds publicUFF selected/terminal物理归属验证；2selectedconvergencewarnings保留|
|P20-D024|S1 / assembly H transformation|CLOSED|assemblycap替换methanolH3/1/charge0；inputsourceH/rollback与表示oracle通过|
|P20-D025|S2 / performance|ACCEPTED_LIMITATION|search/SMARTS/MCA233不利观察含compile1.252×保留；无统计确定globalrate|
|P20-D026|S1 / H representation consumers|CLOSED|T/E/R物理H统一消费者与真实model/pipeline/native范围；field14定制checkpoint需重验|
|P20-D027|S2 / PyG dataset output edge|CLOSED|emptydatasetidentifier用ordinal；named/duplicates/collision保持，43保存graph语义|
|P20-D028|S3 / optional outer-workflow execution|ACCEPTED_LIMITATION|可选training/optimization/realCCDC未验证；命令/资源/查询整合已交付，不宣称可选执行能力|
|P20-D029|S3 / nondefault legacy property queries|CLOSED|nondefaultmedium/solvent查询values/order/duplicates与明确缺失错误；无虚构enrichment|
|P20-D030|S2 / checker portability|CLOSED|CP313/314 AST空fieldchecker归一化；原133失败留档，产品化学未改|
|P20-D031|S1 / CP314 assembly evidence|CLOSED|CP314workerstartupfixture改观测ready后timeout；产品1s默认和startmethod未改|
|P20-D032|S0 / chemical result metadata|CLOSED|SMILES→MOL2中性酸H/Q丢失修复；实际modelmetadata/atomsitejoin；原917失败留档|
|P20-D033|S0 / stereo conversion and geometry consumer|CLOSED|OB/nativebuilder立体到实际O0/Eu/UFF/xTB坐标R,R；原S,S与两complexqualityfail保留|
|P20-D034|S2 / MCA resource-policy boundary|ACCEPTED_LIMITATION|whole-moleculepre-admission阈值8195>8192、resident10241/32768，全部creditsdrained；非strictcap|
|P20-D035|S1 / materialized-H stereo export|CLOSED|materializedH RDKit导出绑定唯一剩余actualgraphH、canonicalorder/parity保持；五ABI受影响关闭|
|P20-D036|S3 / private final-inventory checker|ACCEPTED_LIMITATION|privateinventoryextractor三轮失败、无业务缺陷证据；无第四retry/无PASS|
|P20-D037|S2 / bounded experimental cost|ACCEPTED_LIMITATION|Stage13benzene1.0283/nonaromatic1.2021不利且区间跨1；usablecost非parity/speedup|
|P20-D038|S2 / periodic topology representation|DEFERRED|一般self/parallelperiodicquotient扩展DEFERRED；unsupportedfullprojection保留source并reject，无collapse/扩胞|

因此，仍为 `ACCEPTED_LIMITATION` 的完整ID集合是：D001、D002、D003、D004、D005、D009、D010、D012、D014、D015、D017、D018、D019、D022、D025、D028、D034、D036、D037；唯一 `DEFERRED` 是D038。不是只挑D022/34/36/38呈现风险。所有CLOSED也保留原边界与重新打开条件。

报告核对另外识别 D039（Molecule通用setattr遗留）及 D040（Gaussian缺失热化学值被填零）。
用户授权后已实施局部修复，源码和独立CP311安装验证、原始失败及关闭范围以
[后续修复记录](stages/stage12/final-delivery/validation/postdelivery-interface-cleanup.json)为准。
原38项统计仍是原登记范围；新条目不凭旧验收自动关闭。完整当前状态以[统一索引](deferred_issues.md)为准。

下一工作按授权和状态分层：

| 类别 | 已有范围/状态 | 建议如何表述后续 |
|---|---|---|
| 报告发现的具体缺陷 |D039/D040独立局部修复与验证 |实际结果由新的修复记录归属，历史大套件/模型证据不充当它们的修复证据 |
| 已登记deferred领域扩展 |D038cross-stagegeneralperiodicself/parallelgraph |是有记录的未实现能力，若业务需要须另界定soleindex/writer/native/model设计，不视作本轮已授权实施或旧验收失败 |
| 已接受限制的可选后续优化/诊断 |19个acceptedlimitation，各自reopen触发 |说明实际影响、影响工作与触发条件；优化/更细诊断不是本轮普遍新增验收前提；D036不做第四次retry |
| 另立范围的新科学/平台/模型/算法能力 |独立heldout精度/校准、新平台/driver闭包、全域芳香修复、strictpostadmissioncap、已部署新模型训练或可选researchruntime |属于将来需要明确目标与验证合同的建议，不伪装成当前已批准未完成清单；受影响现有mandatorypath失败则按reopen规则提前处理 |

当前接口/规模只用最终526Pythonfiles/776classes/355dataclasses/8direct+4genericProtocols/2TypedDicts（E01§4），旧Stage12 505/771、31exports是旧快照，当前export与API成员由根代理API inventory审查，不在此重用旧数字。

## 精确证据索引

以下全部为仓库相对路径。JSON定位用字段名/rowid，行号指当前读取字节的字段或表起点；行号不是不可变artifactidentity。对应claim→evidence→scope保存在同目录`delivery_evidence_graph.json`，其中CPU每行另有独立claim与原始summary/参数。

| ID | 仓库相对路径 | 主要字段/行 |
|---|---|---|
|E01|`plan/20_fact_rule_architecture/stages/stage15/integration/report.md`|L63；##5/7/8/9|
|E02|`plan/20_fact_rule_architecture/stages/stage15/integration/validation/checkpoint.json`|L47；final_cp311_source_default/final_installed_matrix/native_and_scientific_evidence|
|E03|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-cpu-performance.json`|L107；rows/families/counts/limits|
|E04|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-gpu-performance.json`|L11；pairs/hardware/limits|
|E05|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-common-test-cost.json`|L10；cells/raw_comparison/hardware_review|
|E06|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-version-matrix.json`|L77；by_version_attempt_wall_medians/cost_disposition|
|E07|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-added-test-cost.json`|L23；cost/original_failed_launch|
|E08|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-new-mechanism-cost.json`|L17；usable_costs/limits|
|E09|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-sdf-write-resources.json`|L414；plain_cost/root_dispositions|
|E10|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-plan12-eu-reference.json`|L69；observations/root_decision/limits|
|E11|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-plan12-eu-uff.json`|L81；observations/per_target_status_counts/limits|
|E12|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-plan12-eu-xtb.json`|L40；route_outcomes/backend/retained_structure_counts|
|E13|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-mca-functional-routing.json`|L6；counts/cases/independent_oracles/repeat_tolerance|
|E14|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/final-repaired-stereo-delivery.json`|L4021；actual_affected_science/limits|
|E15|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/cbond-chemical-reference.json`|L91；chemical_contract/retained_failures|
|E16|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/native-stereo-conversion.json`|L82；actual_models/failure_accounting/limits|
|E17|`plan/20_fact_rule_architecture/stages/stage12/scientific-hardening/validation/report.md`|L28；phosphorus/O0observations|
|E18|`plan/20_fact_rule_architecture/stages/stage12/hydrogen-representation/validation/report.md`|L8；T/E/R与featureprobe|
|E19|`plan/20_fact_rule_architecture/stages/stage13/integration/report.md`|L43；配对成本表/limits|
|E20|`plan/20_fact_rule_architecture/stages/stage14/crystal/validation/report.md`|L74；cost/source/installed/CUDA/failures|
|E21|`plan/20_fact_rule_architecture/stages/stage15/domain/validation/persistence_report.md`|L56；CIFcorpus/completecost/domainlimits|
|E22|`plan/20_fact_rule_architecture/stages/stage15/domain/validation/geometry_report.md`|L56；geometrycost/independentfixtures|
|E23|`plan/20_fact_rule_architecture/stages/stage15/domain/validation/exact_symmetry_report.md`|L15；exactreference/independentfixtures/cost|
|E24|`plan/20_fact_rule_architecture/stages/stage15/domain/validation/linking_report.md`|L13；certifiedlinking/fixtures/cost|
|E25|`plan/20_fact_rule_architecture/deferred_issues.md`|L69；D001–D038|
|E26|`hotpot/cheminfo/operations/inference/mca/MODEL_CARD.md`|L22；modelcard/releaseparity/domain|
|E27|`hotpot/cheminfo/operations/inference/artifacts/data/mca/manifest.json`|L4；artifact/models/property|
|E28|`hotpot/cheminfo/operations/inference/artifacts/data/cbond/manifest.json`|L30；artifact/validation/limits|
|E29|`hotpot/commands/cbond.md`|L103；score/pathweights/thresholds/domain|
|E30|`hotpot/commands/mca.md`|L207；charged/atom/routingdomain|
|E31|`hotpot/cheminfo/rules/physics/forcefields/contracts.py`|L88；StructureAcceptanceThresholds|
|E32|`hotpot/cheminfo/rules/physics/forcefields/acceptance.py`|L33；forcefield/coordinate/bondring/geometrychecks|
|E33|`hotpot/cheminfo/rules/mathematics/geometry/settings.py`|L23；numeric/surfacebudgets|
|E34|`tests/test_cheminfo/forcefields/test_stage12_observed_geometry.py`|L40；contact/rigidtransform/control|
|E35|`plan/20_fact_rule_architecture/stages/stage15/domain/contracts/decisions.md`|L156；exact/numeric/domain|
|E36|`plan/20_fact_rule_architecture/stages/stage15/domain/contracts/observations.md`|L52；metric/maps/finitequerycontrol|
|E37|`plan/12_xtb_workflow/rules/HARD-001-boundary-identity.md`|L14；boundaryidentity|
|E38|`plan/12_xtb_workflow/rules/HARD-002-charge-conservation.md`|L15；chargesums/partialunknown|
|E39|`plan/12_xtb_workflow/rules/HARD-013-backend-electron-uhf-legality.md`|L27；electron/UHFprojectionlegality|
|E40|`plan/12_xtb_workflow/rules/HARD-012-failure-and-gate-semantics.md`|L11；hard/localfinite/recovery|
|E41|`plan/12_xtb_workflow/recovery/RECOVERY-001-legal-state-retry.md`|L10；disabledretry/attribution|
|E42|`plan/20_fact_rule_architecture/stages/stage08/scientific-state/validation/lewis_report.md`|L16；Happlicability/actualpipelineobservations|
|E43|`plan/20_fact_rule_architecture/stages/stage08/forcefields/validation/whole-stage/report.md`|L19；actualmixedpipeline/O0/finitefailures|
|E44|`.plan20-artifacts/20261104/stage12/scientific-hardening/raw/phosphorus/source-attempt02-tools/child.py`|L185；Pangle_sine>1e-3/physicalgraph|
|E45|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/report.md`|L193；main-scope scientific/performance/limits|
|E46|`plan/20_fact_rule_architecture/stages/stage15/domain/validation/generation_report.md`|L32；originalgenerationcost|
|E47|`plan/20_fact_rule_architecture/stages/stage15/domain/validation/relation_report.md`|L43；placementcriteria/cost|
|E48|`plan/20_fact_rule_architecture/stages/stage14/molecule/validation/molecular_report.md`|L36；moleculargeneralCUDA/costfailures|
|E49|`tests/test_cheminfo/kekulize/test_known_limitations.py`|L42；three strictknownlimits|
|E50|`tests/test_cheminfo/test_forcefield_acceptance.py`|L260；independentgatepositive/negative/boundary|
|E51|`plan/20_fact_rule_architecture/stages/stage12/final-delivery/validation/postdelivery-interface-cleanup.json`|D039/D040；actual source/installed commands, observations, failures and root closure|
|E52|`plan/20_fact_rule_architecture/delivery_api_tree.md`|current2c2ef889；526modules/6stubs static declarations；raw inventory under owning local archive|

A01精确成员：`default02/tests/intent.json`、`default02/RESULT.json`、`default02/junit.xml`、`installed01/cp311/tests/intent.json`、`installed01/cp311/RESULT.json`、`installed01/cp311/junit.xml`、`ROOT_REUSE_DECISION.json`、`ROOT_EXTERNAL_IDENTITY.json`、`reuse-review/REUSE.json`、`reuse-review/REVIEW.md`、`ROOT_CHECKER_DECISION.json`、`FINAL_SOURCE_REBIND.json`。原 E01–E50 汇总仅静态抽取并比对counts/字段，没有导入Hotpot或运行计算；D039/D040 的后来实际运行单列在 E51/A02，API 的新静态目录单列在 E52。
