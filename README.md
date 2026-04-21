# Agent Super Memory MVP

基于 **Milvus（向量）+ Neo4j（图）** 的长期记忆服务。对外仅暴露 **两个 HTTP 接口**：用户用自然语言说话，服务端组装系统提示词与工具定义，调用 **Ollama 对话模型** 触发 `store_memory` / `query_memory`，由后端执行真实写入与检索，最后返回模型生成的自然语言回复。

详细数据模型与工具 Schema 见仓库内 [`design.md`](design.md)。

**检索与存储约定（与 design 原文可能略有侧重）**：写入 Milvus 的 `text` 使用中文句式模板，事实的键与值**原样来自** `store_memory` 入参；**事实查询以 Milvus 向量（`ref_type=fact`）为主**，命中后用 `fact_id` 从 Neo4j 回填 `facts`。旧数据若句式不同，可清空集合后重灌。

## 依赖环境

- **Python** 3.10+
- **Milvus** 2.x（默认 `localhost:19530`）
- **Neo4j** 5.x（默认 `bolt://localhost:7687`）
- **Ollama**（默认 `http://localhost:11434`）
  - **对话模型**：用于编排工具调用（默认 `qwen3.5:35b-a3b-q8_0`）
  - **嵌入模型**：写入/检索时向量化（默认 `qwen3-embedding:8b`，向量维默认 **4096**）

## 安装与启动

```bash
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

启动时会连接 Milvus / Neo4j 并初始化集合；若失败，两个对话接口会返回 **503**，日志中有原因说明。

## 配置

支持环境变量或项目根目录 `.env`（`pydantic-settings`）。常用项如下（名称与代码中 `Settings` 字段对应，环境变量为大写蛇形）：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `OLLAMA_CHAT_MODEL` | 对话 / 工具编排模型 | `qwen3.5:35b-a3b-q8_0` |
| `OLLAMA_EMBED_MODEL` | 嵌入模型 | `qwen3-embedding:8b` |
| `OLLAMA_EMBED_DIMENSIONS` | 嵌入输出维度（可选，需与 `VECTOR_DIM` 一致） | 未设置则用模型默认 |
| `VECTOR_DIM` | Milvus 向量字段维度 | `4096` |
| `MILVUS_HOST` / `MILVUS_PORT` / `MILVUS_COLLECTION` | Milvus 连接与集合名 | `localhost` / `19530` / `memory_chunks` |
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | Neo4j | `bolt://localhost:7687` / `neo4j` / `admin123` |
| `DEFAULT_USER_ID` | 两个对话接口使用的用户分区 ID | `default` |

若曾用 **768 维** 建过 `memory_chunks`，与当前默认 **4096** 不一致会导致写入失败，需删除该集合或改用新的 `MILVUS_COLLECTION` 名称。

## HTTP API（仅两个）

请求体均为 JSON，**只有一个字段** `input`（用户原话）。记忆归属用户固定为配置项 **`DEFAULT_USER_ID`**，不由请求体传入。

### 存储

`POST /memory/conversation/store`

```json
{ "input": "记一下：我配偶叫李丽，上周六我们在沃尔玛买了排骨。" }
```

响应示例字段：`reply`（给用户的话）、`tool_called`、`tool_results`（便于调试）。

### 提取（查询）

`POST /memory/conversation/query`

```json
{ "input": "我之前说过我配偶叫什么名字？" }
```

### 测试文件

[`test_main.http`](test_main.http) 中为上述两个接口各提供一条示例请求（适用于 VS Code REST Client / JetBrains HTTP Client）。

## 日志

默认 **INFO** 级别：`main`（请求与生命周期）、`memory_agent`（Ollama 轮次与工具调用）、`memory_service`（连接、Milvus flush、检索分支与结果规模）。异常会打完整堆栈。

## 项目结构（简要）

| 文件 | 作用 |
|------|------|
| `main.py` | FastAPI 入口，仅注册两个对话路由 |
| `config.py` | 配置 |
| `memory_service.py` | Milvus + Neo4j + 嵌入，`store_memory` / `query_memory` |
| `memory_agent.py` | Ollama 多轮对话 + 工具执行编排 |
| `tools_spec.py` | 注入 Ollama 的工具 JSON Schema |
| `design.md` | 系统设计文档 |

## 许可

以项目实际情况为准；未单独声明时仅供内部 / 学习使用。
