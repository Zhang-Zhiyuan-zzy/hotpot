# 当前静态 API 参考

本目录由当前产品快照 `2c2ef889abb6adcceba043b606da2e718dbea109` 的 Python AST 库存生成，日期 `2026-11-05`；未导入 Hotpot、调用可选引擎或执行运行时验证。维护入口为 [完整交付 API 树](../delivery_api_tree.md)，辅助入口为 [辅助模块及类](auxiliary_index.md)。

覆盖 526 个 Python 模块（主交付 354、辅助 172），6 份 native `.pyi`，33 个根包显式导出；主交付包含 418 个非私有本地类、381 个函数、716 个本类成员、1819 个字段与 198 项继承成员。辅助层包含 215 个类、268 个函数、941 个本类成员、863 个字段和 203 项继承成员。

完整签名、property 写权限、字段、别名、源码 owner 与有限本地继承分别列出。显式 `__all__`、直接模块定义和私有实现模块不等价；类值重绑定、实例数据可写不构成批准的化学事实写许可。第三方继承和运行时安装成员未穷尽；参考目录不构成可选模块重新科学验收，也不重试已退役的 D036 检查器。

<a id="raw-inventory"></a>
## 原始库存与按需复现

机器库存约 8.5 MB，按需读取本地忽略证据归档 [原始库存索引](../../../.plan20-artifacts/20261105/overall/delivery-api-inventory/README.md)，其中包含 `api_inventory.json`、逐文件 SHA-256、计数、提取/渲染脚本、发布脚本和文档结构检查结果。维护参考本身提供完整声明导航与成员内容，不依赖原始库存才能阅读。忽略归档不随 Git 分发；缺少本地归档时，先按需取得该归档及其中的原始提取/渲染脚本，再从同一源码快照执行下列命令。缺失归档不构成运行时验证通过。

```bash
python .plan20-artifacts/20261105/overall/delivery-api-inventory/extract_api.py --repo . --out /tmp/hotpot_plan20/short/delivery-api
python .plan20-artifacts/20261105/overall/delivery-api-inventory/render_auxiliary_api.py --inventory /tmp/hotpot_plan20/short/delivery-api/api_inventory.json --out /tmp/hotpot_plan20/short/delivery-aux-api
```

提取器的完整展开中间页 `api_details.md`、`api_signatures.md` 不作为重复的维护入口；本目录按主题分拆的页面提供相同主交付声明，辅助页面补足另外 172 个模块。
