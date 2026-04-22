# Schema 查询：单步（一期）与 ReAct（二期）

## 请求体 `POST /schema/query`

| 字段 | 说明 |
|------|------|
| `query` | 用户自然语言问题 |
| `mode` | **`dry_run`**：不执行 MySQL（单步模式不返回结果行；ReAct 下 `execute_sql` 仅校验）；**`auto_execute`**：执行只读 SQL |
| `min_confidence` | 可选，过滤 Milvus `schema_docs` 检索 |
| `agent_mode` | **`single_pass`**（默认）：一期单步生成 SQL；**`react`**：多步工具循环 |
| `max_steps` | 仅 `react` 生效，最大推理步数（默认 12，上限 40） |

`agent_mode` 与 `mode` 职责分离：**前者**选编排策略，**后者**控制是否连 MySQL 执行 SQL。

## ReAct 行为摘要

- 工具：`search_tables`、`describe_table`、`execute_sql`、`get_foreign_keys`、`final_answer`。
- `get_foreign_keys` 依赖 Neo4j 中 `RELATES_TO`；未导入关系时返回空列表。
- `execute_sql` 在单次请求内最多调用 **8** 次（防止刷库）；仍走 `schema_manager.is_safe_sql` 与只读白名单。
- 响应字段：`final_answer`、`trace`（每步 thought / tool / arguments / observation）、`sql_tool_calls`、`steps_used`。

## 示例

见项目根目录 [`test_sql.http`](../test_sql.http) 中 **Schema-ReAct** 小节。
