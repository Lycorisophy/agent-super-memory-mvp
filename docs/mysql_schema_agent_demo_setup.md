# MySQL 演示库说明（agent_schema_demo）

## 1. 建库与数据

在项目根目录执行（会提示输入密码）：

```bash
mysql -uroot -p < docs/mysql_schema_agent_demo.sql
```

脚本会创建库 **`agent_schema_demo`**，表 **`users`**、**`orders`**，并写入与 `tests/schema_sample_*.md` 一致的示例数据。

## 2. 与应用连接（你本机的 root 账号）

在仓库根目录 `.env` 中配置（**勿将含真实密码的 .env 提交到公开仓库**）：

```ini
MYSQL_URL=mysql+pymysql://root:MY9618sql@127.0.0.1:3306/agent_schema_demo
```

若密码或主机不同，请自行替换 `root`、`MY9618sql`、`127.0.0.1`、库名。

## 3. 与报错「Schema 后端不可用」的关系

- **不是 MySQL 连不上** 时也会出现 Schema 不可用：只要 **Milvus / Neo4j / Ollama** 任一在启动阶段失败，整个记忆后端会失败，`SchemaManager` 也不会初始化。
- 你遇到的 **`NODE KEY` … requires Neo4j Enterprise** 来自 **Neo4j**，与 MySQL 无关。代码已改为使用 **`Column.column_key` 单列 UNIQUE**，可在 **Neo4j 社区版** 上创建约束。请 **重启应用** 后再调 `/schema/ingest`。

若 Neo4j 里曾半失败创建过约束，一般无需处理；若仍有异常，可在 Neo4j Browser 中查看约束列表并删除冲突项后重启服务。

## 4. 生产建议

`auto_execute` 请使用 **仅 SELECT 权限** 的专用账号，不要使用 root。
