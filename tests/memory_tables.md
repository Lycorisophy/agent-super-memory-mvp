## chat_messages（用户和智能体助手的对话记录，智能体对话只存FINAL_ANSWER）
| 字段名 | 类型 | 注释 | 键 | 关联关系 |
|--------|------|------|-----|----------|
| id | BIGINT | 主键，自增ID | PRI | |
| user_id | VARCHAR(128) | 用户标识 | MUL | |
| role | VARCHAR(32) | 说话人身份，用户user 或 智能体assistant | | |
| content | MEDIUMTEXT | 说话内容 | | |
| created_at | DATETIME | 创建时间 | | CURRENT_TIMESTAMP |

## permanent_memory（永久记忆表）
| 字段名 | 类型 | 注释 | 键 | 关联关系 |
|--------|------|------|-----|----------|
| user_id | VARCHAR(128) | 用户标识 | PRI（复合主键之一） | |
| category | VARCHAR(32) | 记忆分类 | PRI（复合主键之一） | |
| content | VARCHAR(1000) | 记忆内容 | | '' |
| updated_at | DATETIME | 更新时间 | | CURRENT_TIMESTAMP（自动更新） |