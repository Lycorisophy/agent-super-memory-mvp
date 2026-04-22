# 永驻记忆表（MySQL）

每条最多 **1000 字**；按 `user_id` + `category` 唯一一行。`category` 取值为 `user_identity` / `agent_personality` / `work_norms`（与工具入参中文「用户身份」「智能体性格」「工作规范」对应）。

## DDL 示例

```sql
CREATE TABLE permanent_memory (
  user_id VARCHAR(128) NOT NULL,
  category VARCHAR(32) NOT NULL,
  content VARCHAR(1000) NOT NULL DEFAULT '',
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (user_id, category),
  KEY idx_pm_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## 权限

与 [`dialogue_table_ddl.md`](dialogue_table_ddl.md) 相同连接：`MYSQL_DIALOGUE_URL` 或回退 `MYSQL_URL`；账号需 **SELECT、INSERT、UPDATE**。

## 测试用初始化数据（三条 SQL）

`user_id` 与接口默认一致（`DEFAULT_USER_ID`，一般为 `default`）。若已存在相同主键，请先 `DELETE` 再执行，或改用 `ON DUPLICATE KEY UPDATE`。

```sql
INSERT INTO permanent_memory (user_id, category, content, updated_at) VALUES ('default', 'user_identity', '【测试】用户身份：本地联调账号，非生产数据。', NOW());
```

```sql
INSERT INTO permanent_memory (user_id, category, content, updated_at) VALUES ('default', 'agent_personality', '【测试】智能体性格：回答简洁、先结论后细节。', NOW());
```

```sql
INSERT INTO permanent_memory (user_id, category, content, updated_at) VALUES ('default', 'work_norms', '【测试】工作规范：不臆造接口字段；不确定时先说明假设。', NOW());
```
