# Agent Super Memory MVP

基于 **Milvus（向量）+ Neo4j（图）** 的长期记忆服务。实现与 **[design.md](docs/长期记忆系统详细设计文档 v4.0.md) v4.0** 及 **[长期记忆系统 v5.0 增量修改文档（相对 v4.0）.md](长期记忆系统%20v5.0%20增量修改文档（相对%20v4.0）.md)** 对齐：**单一 `:Memory` 节点**、**统一中文模板 `content`**；工具 **`memories` + `relations`（temp_id）** 与 **`query_text` + `memory_types`**；v5 起 Neo4j 侧支持可选 **`tense` / `confidence`** 及查询过滤（Milvus 结构不变）。

对外暴露 **记忆对话两个 HTTP 接口**（`store` / `query`），以及可选的 **统一对话 ReAct 接口**（见下）：用户自然语言 → 服务端组装系统提示词与工具定义 → **Ollama** 触发工具 → 后端写入/检索 → 返回模型自然语言回复。

## v4 与旧版不兼容说明

- **Milvus**：集合字段为 `memory_id`、`user_id`、`text`、`embedding`、`timestamp`、`memory_type`（默认集合名 **`memory_vectors`**）。旧版 `memory_chunks`（`chunk_id`/`ref_type`/`ref_id` 等）**不可混用**，请删旧集合或换新 `MILVUS_COLLECTION`。
- **Neo4j**：使用 `:Memory` + `[:HAS_MEMORY]`；旧版 `:Event`/`:Fact`/`:Knowledge` 相关约束与数据不再由本代码写入。
- **嵌入**：仓库默认 **`qwen3-embedding:8b`** 与 **`VECTOR_DIM=4096`**（与 `config.py` 一致）。若改用 `nomic-embed-text` 等其它模型，必须在 `.env` 中同时修改 `OLLAMA_EMBED_MODEL` 与 `VECTOR_DIM`（并重建或更换 Milvus 集合）。

## v5.0 增量（相对 v4）

- **Neo4j `Memory`**：可选属性 `tense`（`past` / `present` / `future`）、`confidence`（`real` / `imagined` / `planned`）。v4 已存节点无该属性视为 `null`，**不传查询过滤时行为与 v4 一致**。
- **`store_memory`**：`memories[]` 每项可带 `tense`、`confidence`。
- **`query_memory`**：可选参数 `tense`、`confidence`，在 Neo4j 装配阶段过滤；传过滤时 Milvus 候选数会自动放大，减少过滤后条数不足。
- **返回**：`memories` 每项在节点有值时附带 `tense` / `confidence` 字段。

## 依赖环境

- **Python** 3.10+
- **Milvus** 2.x（默认 `localhost:19530`）
- **Neo4j** 5.x（默认 `bolt://localhost:7687`）
- **Ollama**
  - **对话模型**（默认 `qwen3.5:35b-a3b-q8_0`，由 `OLLAMA_CHAT_MODEL` 配置）
  - **嵌入模型**（默认 `qwen3-embedding:8b`，**4096** 维，与 `VECTOR_DIM` 一致）

## 安装与启动

```bash
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

启动时会连接 Milvus / Neo4j 并初始化集合；若失败，记忆对话接口会返回 **503**。若另行配置对话 MySQL 失败，仅 **`POST /memory/conversation/unified`** 返回 **503**，不影响 `store` / `query`。

### 最小自检（无外部服务）

```bash
python tests/test_v4_memory_unit.py
python tests/test_v5_enums.py
python -m pytest tests/test_dialogue_store_unit.py tests/test_unified_dialogue_agent.py -q
```

## 配置

支持环境变量或项目根目录 `.env`。常用项：

| 环境变量 | 说明 | 默认值 |
|----------|------|----------------|
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `OLLAMA_CHAT_MODEL` | 对话 / 工具编排 | `qwen3.5:35b-a3b-q8_0` |
| `OLLAMA_EMBED_MODEL` | 嵌入模型 | `qwen3-embedding:8b` |
| `OLLAMA_EMBED_DIMENSIONS` | 嵌入输出维度（须与向量维一致） | 未设置 |
| `VECTOR_DIM` | Milvus 向量维度 | `4096` |
| `MILVUS_COLLECTION` | 集合名 | `memory_vectors` |
| `NEO4J_*` | Neo4j 连接 | 见 `config.py` |
| `DEFAULT_USER_ID` | 对话接口使用的用户分区 | `default` |
| `MYSQL_DIALOGUE_URL` | 统一对话：对话表连接串（**SELECT+INSERT**）；空则沿用 `MYSQL_URL` | 空 |
| `DIALOGUE_TABLE` | 对话表名 | `chat_messages` |
| `DIALOGUE_COL_*` | 列名：`ID` / `USER_ID` / `ROLE` / `CONTENT` / `CREATED_AT` | 见 `config.py` |
| `DIALOGUE_FETCH_LIMIT` | 拉取最近条数 | `10` |
| `DIALOGUE_OLDER_FETCH_MAX` | `fetch_older_chat` 单次上限 | `20` |
| `UNIFIED_DIALOGUE_MAX_STEPS` | 统一 ReAct 默认最大步数 | `5` |
| `UNIFIED_DIALOGUE_MAX_STEPS_CAP` | 统一 ReAct 步数硬上限（接口 `max_steps` 不超过此值） | `8` |
| `PERMANENT_MEMORY_TABLE` | 永驻记忆表（与对话库同一连接；**SELECT+INSERT+UPDATE**） | `permanent_memory` |

建表与权限说明见 [`docs/dialogue_table_ddl.md`](docs/dialogue_table_ddl.md)、[`docs/permanent_memory_ddl.md`](docs/permanent_memory_ddl.md)。永驻记忆在 **`/memory/conversation/store`**、**`/query`**、**`/unified`** 中注入系统提示；可用工具 **`update_permanent_memory`**（三类中文类别 + 至多 1000 字）。

## HTTP API

### 记忆对话（两接口）

请求体 JSON 含 **`input`**。`user_id` 固定为 **`DEFAULT_USER_ID`**。

- `POST /memory/conversation/store` — 写入记忆（统一模板）
- `POST /memory/conversation/query` — 语义检索

### 统一对话 ReAct（单接口）

- `POST /memory/conversation/unified` — 请求体：`input`（本轮）、可选 `include_trace`、`max_steps`；从 MySQL 读最近 `DIALOGUE_FETCH_LIMIT` 条上下文，内部 ReAct 组合 `store_memory` / `query_memory` / `fetch_older_chat`（若永驻库可用则另有 `update_permanent_memory`）；同步返回 `final_answer`；**本轮 user/assistant 两行在响应后由后台任务异步写入**（失败仅打日志）。

详见 [`test_main.http`](test_main.http)。

## 项目结构

| 文件 | 作用 |
|------|------|
| `main.py` | FastAPI 入口（含 `/memory/conversation/unified`） |
| `dialogue_store.py` | MySQL 对话表只读/写入 |
| `unified_dialogue_agent.py` | 统一对话 JSON ReAct |
| `permanent_memory_store.py` | MySQL 永驻记忆读/写 |
| `config.py` | 配置 |
| `memory_service.py` | v4 Milvus + Neo4j + 嵌入 |
| `memory_agent.py` | Ollama 编排 + design §6.1/§6.2 提示词 |
| `tools_spec.py` | 工具 JSON Schema（v4 + v5） |
| `design.md` | 系统设计 v4 |

## 许可

以项目实际情况为准；未单独声明时仅供内部 / 学习使用。
