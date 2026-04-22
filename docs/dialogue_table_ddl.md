# 统一对话接口：MySQL 对话表 DDL 与权限

`/memory/conversation/unified` 会在启动时连接对话库，读取最近 `DIALOGUE_FETCH_LIMIT` 条历史，并在响应后异步写入本轮 **user** 与 **assistant** 各一行。表结构建议与业务一致后，再通过 `POST /schema/ingest` 录入元数据，便于模型理解字段含义；**实际查询仅由服务端参数化 SQL 完成**，模型不能对对话表执行任意 `execute_sql`。

## 建表示例

```sql
CREATE TABLE `chat_messages` (
  `id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键，自增ID',
  `user_id` varchar(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '用户标识',
  `role` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '说话人身份，用户user 或 智能体assistant',
  `content` mediumtext CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci COMMENT '说话内容',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  KEY `idx_chat_user_id` (`user_id`,`id` DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='用户和智能体助手的对话记录，智能体对话只存FINAL_ANSWER';
```

- 排序与游标：默认按配置列 `dialogue_col_id`（上例 `id`）降序取「最近 N 条」，再反转为时间正序拼入提示词；`fetch_older_chat` 使用 `id < before_id` 拉取更早消息。
- 若不用自增主键，可改用 `created_at` 排序（需自行调整代码中的排序列逻辑）；当前实现以 **`id` 为数值游标** 为前提。

## 数据库账号权限

| 用途 | 建议 |
|------|------|
| 仅 Schema 助手只读 | `mysql_url` 使用只读账号 |
| 统一对话读写 | `mysql_dialogue_url` 使用具备 **SELECT、INSERT** 的账号；可与 `mysql_url` 同库，但只读账号会导致后台写入失败 |

环境变量见根目录 `README.md` 配置表（`MYSQL_DIALOGUE_URL`、`DIALOGUE_TABLE`、列名映射等）。

## 可选集成验证（CI 外）

本地 Docker 起 MySQL、执行上文 DDL、配置 `.env` 后，可调用 `POST /memory/conversation/unified`（`{"input":"…"}`）做一次端到端验证；单测见 `tests/test_dialogue_store_unit.py` 与 `tests/test_unified_dialogue_agent.py`（mock，无外部依赖）。
