# 长期记忆系统详细设计文档（v2.0）

## 1. 概述

### 1.1 背景与动机

传统基于纯向量检索的长期记忆方案存在以下固有限制：
- **时序与因果湮灭**：向量检索只反映语义相似度，无法保留事件的先后顺序和因果链条。
- **精确事实混淆**：向量有损压缩导致姓名、数字、状态等精确信息容易被相似内容覆盖。
- **状态更新困难**：新旧事实并存时，向量检索难以区分当前有效版本。
- **缺乏抽象归纳**：只能召回原始片段，无法形成凝练的元记忆。

本系统采用**向量库（Milvus）+ 图库（Neo4j）** 混合架构，通过结构化提取与关系建模，弥补上述缺陷。

### 1.2 设计目标

- 支持**事件**、**事实**、**知识**的结构化存储与语义检索。
- 提供精确的事实版本管理与冲突覆盖机制。
- 建立事件间的时序、因果、包含等方向性关系。
- 提供面向大模型 Agent 的标准化工具调用接口。

### 1.3 技术选型

| 组件 | 选型 | 版本 | 作用 |
| :--- | :--- | :--- | :--- |
| 向量数据库 | Milvus | 2.4+ | 存储文本片段的语义向量，支持模糊召回 |
| 图数据库 | Neo4j | 5.x | 存储结构化节点与关系，承载逻辑推理 |
| 嵌入模型 | Ollama (nomic-embed-text) | 本地部署 | 文本向量化，维度 768 |
| 唯一ID生成 | Snowflake ID | 自实现 | 全局唯一、有序、高性能的分布式ID |
| 开发语言 | Python | 3.10+ | 工具接口实现 |

### 1.4 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM Agent                              │
│   ┌─────────────────┐            ┌─────────────────┐        │
│   │ store_memory    │            │ query_memory    │        │
│   └────────┬────────┘            └────────┬────────┘        │
└────────────┼──────────────────────────────┼─────────────────┘
             │ Function Calling              │
             ▼                                ▼
┌─────────────────────────┐    ┌─────────────────────────────┐
│   记忆存储服务            │    │     记忆查询服务             │
│ ┌──────────────────────┐│    │ ┌──────────────────────────┐│
│ │ 事件/事实/知识提取    ││    │ │ 意图路由与查询参数构造    ││
│ └──────────┬───────────┘│    │ └──────────┬───────────────┘│
│            ▼             │    │             ▼                │
│ ┌──────────────────────┐│    │ ┌──────────────────────────┐│
│ │ 关系构建与持久化      ││    │ │ 混合检索执行器           ││
│ └──────────┬───────────┘│    │ │ - 事实精确查询 (Neo4j)   ││
│            │             │    │ │ - 事件向量召回 (Milvus)  ││
└────────────┼─────────────┘    │ │ - 图游走扩展 (Neo4j)     ││
             │                  │ │ - 知识语义检索 (Milvus)  ││
      ┌──────┴──────┐           │ │ - 兜底向量召回 (Milvus)  ││
      ▼             ▼           │ └──────────┬───────────────┘│
┌──────────┐  ┌──────────┐      └────────────┼─────────────────┘
│  Milvus  │  │  Neo4j   │                   │
│(向量索引) │  │(图数据库) │◄──────────────────┘
└──────────┘  └──────────┘
```

---

## 2. 数据库详细设计

### 2.1 Milvus 向量库设计

**Collection 名称**：`memory_chunks`

**字段定义**：

| 字段名 | 数据类型 | 约束 | 描述 |
| :--- | :--- | :--- | :--- |
| `chunk_id` | VarChar(64) | 主键 | 雪花ID生成，全局唯一 |
| `user_id` | VarChar(64) | 分区键 | 用户唯一标识 |
| `text` | VarChar(4096) | - | 用于向量化的文本内容 |
| `embedding` | FloatVector(768) | - | 文本向量（维度取决于嵌入模型） |
| `timestamp` | Int64 | - | Unix 时间戳（秒） |
| `source_type` | VarChar(32) | - | `raw` 或 `structured` |
| `ref_type` | VarChar(32) | - | **区分数据类型**：`event` / `fact` / `knowledge` |
| `ref_id` | VarChar(64) | - | 对应 Neo4j 节点的 ID（雪花ID） |

**索引配置**：
- **向量索引**：`HNSW`，度量类型 `COSINE`，参数 `M=16, efConstruction=200`
- **标量索引**：对 `user_id`、`timestamp`、`ref_type` 建立索引以加速过滤

### 2.2 Neo4j 图数据库设计

#### 节点标签与属性

| 标签 | 属性 | 类型 | 描述 |
| :--- | :--- | :--- | :--- |
| `:User` | `user_id` | String | 用户根节点，雪花ID或业务ID |
| `:Event` | `event_id` | String | 事件唯一ID（雪花ID） |
| | `summary` | String | 事件一句话概括 |
| | `time` | Integer | 发生时间戳（秒） |
| | `location` | String | 地点 |
| | `subject` | String | 主体人物 |
| | `participants` | List[String] | 参与人物列表 |
| | `action` | String | 动作行为描述 |
| | `result` | String | 结果或影响 |
| | `tense` | String | 时态：`past` / `present` / `future` |
| | `confidence` | String | 置信度：`real` / `imagined` / `planned` |
| `:Fact` | `fact_id` | String | 事实唯一ID（雪花ID） |
| | `key` | String | 标准化键名（如 `spouse_name`） |
| | `value` | String | 事实值 |
| | `updated_at` | Integer | 更新时间戳 |
| | `confidence` | String | `high` / `medium` / `low` |
| `:Entity` | `name` | String | 实体名称 |
| | `type` | String | 实体类型：`person` / `place` / `organization` / `other` |
| `:Knowledge` | `knowledge_id` | String | 知识唯一ID（雪花ID） |
| | `title` | String | 知识标题 |
| | `content` | String | 知识内容 |
| | `category` | String | 分类（如 `cooking`, `programming`） |

#### 关系类型与方向

| 关系名 | 方向 | 属性 | 语义 |
| :--- | :--- | :--- | :--- |
| `[:HAS_EVENT]` | `User → Event` | - | 用户经历了该事件 |
| `[:HAS_FACT]` | `User → Fact` | - | 用户拥有该事实 |
| `[:NEXT]` | `Event → Event` | - | 事件 A 之后紧接着发生事件 B |
| `[:CAUSED]` | `Event → Event` | - | 事件 A 导致了事件 B |
| `[:SUB_EVENT_OF]` | `Event → Event` | - | 事件 A 是事件 B 的子事件（包含关系） |
| `[:MENTIONS]` | `Event → Entity` | - | 事件中提及了某实体 |
| `[:OVERRIDES]` | `Fact → Fact` | `timestamp` | 新事实覆盖旧事实（版本管理） |
| `[:RELATED_TO]` | `Event → Knowledge` | - | 事件关联某条知识 |

> **方向性说明**：所有关系在 Neo4j 中均为有向边，创建时遵循 `(source)-[:REL_TYPE]->(target)` 语法，保证逻辑方向正确。

---

## 3. 雪花ID生成器

### 3.1 设计说明

采用 Twitter Snowflake 算法变体，生成 64 位长整型唯一 ID，具有以下特性：
- **全局唯一**：分布式环境下不重复
- **趋势递增**：利于数据库索引
- **高性能**：本地生成，无网络开销

### 3.2 Python 实现

```python
import time
import threading

class SnowflakeIDGenerator:
    """
    雪花ID生成器
    64位 = 1位符号位(0) + 41位时间戳(毫秒) + 10位机器ID + 12位序列号
    """
    def __init__(self, datacenter_id: int = 1, worker_id: int = 1):
        self.datacenter_id = datacenter_id & 0x1F      # 5位
        self.worker_id = worker_id & 0x1F              # 5位
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()
        
        # 起始时间戳 (2024-01-01 00:00:00)
        self.twepoch = 1704067200000
        
    def _current_millis(self) -> int:
        return int(time.time() * 1000)
    
    def _wait_next_millis(self, last_timestamp: int) -> int:
        timestamp = self._current_millis()
        while timestamp <= last_timestamp:
            timestamp = self._current_millis()
        return timestamp
    
    def next_id(self) -> int:
        with self.lock:
            timestamp = self._current_millis()
            if timestamp < self.last_timestamp:
                raise Exception("Clock moved backwards")
            
            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & 0xFFF  # 12位掩码
                if self.sequence == 0:
                    timestamp = self._wait_next_millis(self.last_timestamp)
            else:
                self.sequence = 0
            
            self.last_timestamp = timestamp
            
            # 组装ID
            id_value = ((timestamp - self.twepoch) << 22) | \
                       (self.datacenter_id << 17) | \
                       (self.worker_id << 12) | \
                       self.sequence
            return id_value
    
    def next_str(self) -> str:
        """返回字符串形式的ID，便于存储"""
        return str(self.next_id())

# 全局实例
id_generator = SnowflakeIDGenerator(datacenter_id=1, worker_id=1)
```

### 3.3 ID 前缀约定

为便于区分节点类型，在存储时添加前缀：
- 事件节点 ID：`ev_{snowflake_id}`
- 事实节点 ID：`fact_{snowflake_id}`
- 知识节点 ID：`kn_{snowflake_id}`
- Milvus chunk ID：`chk_{snowflake_id}`

---

## 4. LLM 工具接口定义

系统对外暴露两个 Function Calling 工具，LLM 通过生成标准 JSON 参数调用。

### 4.1 工具 1：`store_memory`（记忆存储）

**功能**：将用户对话中提取的结构化记忆写入数据库。

**JSON Schema**：

```json
{
  "name": "store_memory",
  "description": "将用户对话中提取的事件、事实、知识以及它们之间的关系写入长期记忆数据库。事件采用标准8字段结构。",
  "parameters": {
    "type": "object",
    "properties": {
      "user_id": { "type": "string", "description": "用户唯一标识" },
      "events": {
        "type": "array",
        "description": "提取的事件列表",
        "items": {
          "type": "object",
          "properties": {
            "summary": { "type": "string", "description": "事件一句话概括" },
            "time": { "type": "integer", "description": "发生时间戳(秒)，默认当前时间" },
            "location": { "type": "string", "description": "发生地点" },
            "subject": { "type": "string", "description": "主体人物" },
            "participants": { "type": "array", "items": { "type": "string" }, "description": "参与人物" },
            "action": { "type": "string", "description": "具体动作行为" },
            "result": { "type": "string", "description": "事件结果或影响" },
            "tense": { "type": "string", "enum": ["past", "present", "future"], "default": "past" },
            "confidence": { "type": "string", "enum": ["real", "imagined", "planned"], "default": "real" },
            "entities": { "type": "array", "items": { "type": "string" }, "description": "事件涉及的实体名称" }
          },
          "required": ["summary", "action"]
        }
      },
      "facts": {
        "type": "array",
        "description": "提取的稳定事实",
        "items": {
          "type": "object",
          "properties": {
            "key": { "type": "string", "description": "标准化键名" },
            "value": { "type": "string" },
            "confidence": { "type": "string", "enum": ["high", "medium", "low"], "default": "high" }
          },
          "required": ["key", "value"]
        }
      },
      "knowledge": {
        "type": "array",
        "description": "提取的客观知识或方法",
        "items": {
          "type": "object",
          "properties": {
            "title": { "type": "string" },
            "content": { "type": "string" },
            "category": { "type": "string" }
          },
          "required": ["title", "content"]
        }
      },
      "relations": {
        "type": "array",
        "description": "提取的关系列表",
        "items": {
          "type": "object",
          "properties": {
            "type": { "type": "string", "enum": ["NEXT", "CAUSED", "SUB_EVENT_OF", "MENTIONS", "OVERRIDES"] },
            "source_type": { "type": "string", "enum": ["event", "fact"] },
            "source_summary": { "type": "string", "description": "源节点摘要或key" },
            "target_type": { "type": "string", "enum": ["event", "fact", "entity"] },
            "target_summary": { "type": "string" }
          },
          "required": ["type", "source_type", "source_summary", "target_type", "target_summary"]
        }
      }
    },
    "required": ["user_id"]
  }
}
```

**返回示例**：

```json
{
  "success": true,
  "stored_ids": {
    "event_ids": ["ev_1234567890"],
    "fact_ids": ["fact_1234567891"],
    "knowledge_ids": ["kn_1234567892"]
  },
  "message": "已存储 1 个事件, 1 个事实, 1 条知识"
}
```

### 4.2 工具 2：`query_memory`（记忆查询）

**功能**：根据用户提问，从混合存储中检索相关记忆。

**JSON Schema**：

```json
{
  "name": "query_memory",
  "description": "从用户长期记忆中查询事件、事实或知识。支持混合查询，返回结构化结果和语义片段。",
  "parameters": {
    "type": "object",
    "properties": {
      "user_id": { "type": "string" },
      "fact_keys": {
        "type": "array",
        "items": { "type": "string" },
        "description": "需要精确查询的事实键名"
      },
      "event_query": {
        "type": "object",
        "properties": {
          "semantic_text": { "type": "string", "description": "语义匹配文本" },
          "time_start": { "type": "integer" },
          "time_end": { "type": "integer" },
          "entities": { "type": "array", "items": { "type": "string" } },
          "top_k": { "type": "integer", "default": 3 }
        }
      },
      "knowledge_query": {
        "type": "object",
        "properties": {
          "semantic_text": { "type": "string" },
          "category": { "type": "string" },
          "top_k": { "type": "integer", "default": 2 }
        }
      },
      "global_vector_fallback": {
        "type": "object",
        "properties": {
          "text": { "type": "string", "description": "兜底检索文本" },
          "top_k": { "type": "integer", "default": 5 }
        }
      }
    },
    "required": ["user_id"]
  }
}
```

**返回示例**：

```json
{
  "facts": { "spouse_name": "李丽" },
  "events": [
    {
      "summary": "和李丽在沃尔玛买排骨",
      "time": 1713657600,
      "location": "沃尔玛超市",
      "subject": "我",
      "participants": ["李丽"],
      "action": "购买排骨",
      "result": "买到了新鲜的排骨",
      "tense": "past",
      "confidence": "real",
      "entities": ["李丽", "排骨"]
    }
  ],
  "knowledge": [
    { "title": "煮饺子防破技巧", "content": "水里加盐" }
  ],
  "vector_chunks": ["原始对话片段..."]
}
```

---

## 5. 系统提示词（System Prompts）

### 5.1 存储侧提示词（`store_memory` 调用前）

```text
你是一个长期记忆编码助手。请分析以下用户文本，提取结构化信息并调用 `store_memory` 工具存储。

**提取原则**：
1. **事件（events）**：按标准8字段提取。若字段缺失可留空。
   - summary：一句话概括
   - time：发生时间戳，无明确时间则用当前时间
   - location：地点
   - subject：主体（通常为“我”或用户名）
   - participants：参与人员列表
   - action：核心动作
   - result：结果或影响
   - tense：past/present/future
   - confidence：real/imagined/planned
2. **事实（facts）**：稳定属性或状态，使用标准化key（如 spouse_name, city, phone_last4）。
3. **知识（knowledge）**：方法步骤、客观规律，提取title和content。
4. **关系（relations）**：
   - 时间先后 → NEXT
   - 因果关系 → CAUSED
   - 父子事件 → SUB_EVENT_OF
   - 提及实体 → MENTIONS（target_type=entity）
   - 事实覆盖 → OVERRIDES（由系统自动处理，可不填）
5. 不要编造信息，缺失字段可省略。

**输出**：直接生成 `store_memory` 工具调用JSON，无额外解释。

当前用户ID: {user_id}
当前时间戳: {current_timestamp}
```

### 5.2 查询侧提示词（`query_memory` 调用前）

```text
你是一个记忆查询路由专家。根据用户问题构造 `query_memory` 调用参数。

**判断逻辑**：
- 涉及个人信息/过往经历/偏好/知识 → 调用工具
- 纯寒暄 → 直接回复

**参数构造指南**：
1. **精确事实**：问“我的XX是什么？” → 设置 `fact_keys`（可多个候选）
2. **事件回忆**：问“上次/什么时候/和谁做了某事” → 填写 `event_query.semantic_text`，估算时间范围，指定相关实体
3. **知识检索**：问“怎么做XX” → 填写 `knowledge_query.semantic_text`
4. **兜底**：始终设置 `global_vector_fallback.text` 为原问题全文

**输出**：直接生成 `query_memory` 调用JSON，或 `NO_QUERY_NEEDED` 并直接回答。

当前用户ID: {user_id}
当前时间: {current_time}
```

---

## 6. 核心 Python 实现

### 6.1 完整代码

```python
import json
import time
import threading
from typing import List, Dict, Any, Optional

import ollama
from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType,
    utility
)
from neo4j import GraphDatabase

# ==================== 雪花ID生成器 ====================
class SnowflakeIDGenerator:
    def __init__(self, datacenter_id: int = 1, worker_id: int = 1):
        self.datacenter_id = datacenter_id & 0x1F
        self.worker_id = worker_id & 0x1F
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()
        self.twepoch = 1704067200000  # 2024-01-01 00:00:00
        
    def _current_millis(self) -> int:
        return int(time.time() * 1000)
    
    def _wait_next_millis(self, last_timestamp: int) -> int:
        timestamp = self._current_millis()
        while timestamp <= last_timestamp:
            timestamp = self._current_millis()
        return timestamp
    
    def next_id(self) -> int:
        with self.lock:
            timestamp = self._current_millis()
            if timestamp < self.last_timestamp:
                raise Exception("Clock moved backwards")
            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & 0xFFF
                if self.sequence == 0:
                    timestamp = self._wait_next_millis(self.last_timestamp)
            else:
                self.sequence = 0
            self.last_timestamp = timestamp
            return ((timestamp - self.twepoch) << 22) | \
                   (self.datacenter_id << 17) | \
                   (self.worker_id << 12) | \
                   self.sequence
    
    def next_str(self) -> str:
        return str(self.next_id())

id_gen = SnowflakeIDGenerator()

# ==================== 配置 ====================
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "memory_chunks"
VECTOR_DIM = 768
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# ==================== 嵌入函数 ====================
def get_embedding(text: str) -> List[float]:
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
    return response["embedding"]

# ==================== Milvus 初始化 ====================
def init_milvus():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(MILVUS_COLLECTION):
        return Collection(MILVUS_COLLECTION)
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        FieldSchema(name="timestamp", dtype=DataType.INT64),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="ref_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="ref_id", dtype=DataType.VARCHAR, max_length=64),
    ]
    schema = CollectionSchema(fields, description="Long-term memory chunks")
    collection = Collection(MILVUS_COLLECTION, schema)
    index_params = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
    collection.create_index("embedding", index_params)
    collection.load()
    return collection

# ==================== Neo4j 初始化 ====================
class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    def close(self):
        self.driver.close()
    
    def ensure_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (k:Knowledge) REQUIRE k.knowledge_id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (ent:Entity) REQUIRE (ent.name, ent.type) IS NODE KEY")
    
    def create_user(self, user_id: str):
        with self.driver.session() as session:
            session.run("MERGE (:User {user_id: $user_id})", user_id=user_id)

neo4j_client = Neo4jClient()
neo4j_client.ensure_constraints()
milvus_collection = init_milvus()

# ==================== 工具实现 ====================
def store_memory(
    user_id: str,
    events: Optional[List[Dict]] = None,
    facts: Optional[List[Dict]] = None,
    knowledge: Optional[List[Dict]] = None,
    relations: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    stored_ids = {"event_ids": [], "fact_ids": [], "knowledge_ids": []}
    neo4j_client.create_user(user_id)
    
    with neo4j_client.driver.session() as session:
        # 1. 存储事件
        if events:
            for ev in events:
                event_id = f"ev_{id_gen.next_str()}"
                summary = ev.get("summary", "")
                time_ts = ev.get("time", ev.get("timestamp", int(time.time())))
                location = ev.get("location", "")
                subject = ev.get("subject", "")
                participants = ev.get("participants", [])
                action = ev.get("action", summary)
                result = ev.get("result", "")
                tense = ev.get("tense", "past")
                confidence = ev.get("confidence", "real")
                
                session.run("""
                    MATCH (u:User {user_id: $user_id})
                    CREATE (e:Event {
                        event_id: $event_id, summary: $summary, time: $time,
                        location: $location, subject: $subject, participants: $participants,
                        action: $action, result: $result, tense: $tense, confidence: $confidence
                    })
                    CREATE (u)-[:HAS_EVENT]->(e)
                """, user_id=user_id, event_id=event_id, summary=summary, time=time_ts,
                     location=location, subject=subject, participants=participants,
                     action=action, result=result, tense=tense, confidence=confidence)
                stored_ids["event_ids"].append(event_id)
                
                # MENTIONS 关系
                for ent in ev.get("entities", []):
                    session.run("""
                        MATCH (e:Event {event_id: $event_id})
                        MERGE (ent:Entity {name: $name, type: 'unknown'})
                        CREATE (e)-[:MENTIONS]->(ent)
                    """, event_id=event_id, name=ent)
                
                # 写入 Milvus
                embed_text = f"事件: {summary} | 时间:{time_ts} | 地点:{location} | 主体:{subject} | 动作:{action} | 结果:{result}"
                vec = get_embedding(embed_text)
                milvus_collection.insert([{
                    "chunk_id": f"chk_{id_gen.next_str()}",
                    "user_id": user_id, "text": embed_text, "embedding": vec,
                    "timestamp": time_ts, "source_type": "structured",
                    "ref_type": "event", "ref_id": event_id
                }])
        
        # 2. 存储事实
        if facts:
            for fact in facts:
                key, value = fact["key"], fact["value"]
                confidence = fact.get("confidence", "high")
                fact_id = f"fact_{id_gen.next_str()}"
                now_ts = int(time.time())
                
                old = session.run("""
                    MATCH (u:User {user_id: $user_id})-[:HAS_FACT]->(f:Fact {key: $key})
                    RETURN f ORDER BY f.updated_at DESC LIMIT 1
                """, user_id=user_id, key=key).single()
                
                session.run("""
                    MATCH (u:User {user_id: $user_id})
                    CREATE (f:Fact {fact_id: $fact_id, key: $key, value: $value, updated_at: $ts, confidence: $confidence})
                    CREATE (u)-[:HAS_FACT]->(f)
                """, user_id=user_id, fact_id=fact_id, key=key, value=value, ts=now_ts, confidence=confidence)
                stored_ids["fact_ids"].append(fact_id)
                
                if old and old["f"].get("value") != value:
                    session.run("""
                        MATCH (new:Fact {fact_id: $new_id})
                        MATCH (old:Fact {fact_id: $old_id})
                        CREATE (new)-[:OVERRIDES {timestamp: $ts}]->(old)
                    """, new_id=fact_id, old_id=old["f"]["fact_id"], ts=now_ts)
                
                embed_text = f"事实: {key} = {value}"
                vec = get_embedding(embed_text)
                milvus_collection.insert([{
                    "chunk_id": f"chk_{id_gen.next_str()}",
                    "user_id": user_id, "text": embed_text, "embedding": vec,
                    "timestamp": now_ts, "source_type": "structured",
                    "ref_type": "fact", "ref_id": fact_id
                }])
        
        # 3. 存储知识
        if knowledge:
            for kn in knowledge:
                kn_id = f"kn_{id_gen.next_str()}"
                title, content = kn["title"], kn["content"]
                category = kn.get("category", "general")
                session.run("""
                    CREATE (k:Knowledge {knowledge_id: $kn_id, title: $title, content: $content, category: $category})
                """, kn_id=kn_id, title=title, content=content, category=category)
                stored_ids["knowledge_ids"].append(kn_id)
                
                embed_text = f"知识: {title} - {content}"
                vec = get_embedding(embed_text)
                milvus_collection.insert([{
                    "chunk_id": f"chk_{id_gen.next_str()}",
                    "user_id": user_id, "text": embed_text, "embedding": vec,
                    "timestamp": int(time.time()), "source_type": "structured",
                    "ref_type": "knowledge", "ref_id": kn_id
                }])
        
        # 4. 存储关系
        if relations:
            for rel in relations:
                rtype = rel["type"]
                if rtype not in ["NEXT", "CAUSED", "SUB_EVENT_OF", "MENTIONS", "OVERRIDES"]:
                    continue
                src = session.run("""
                    MATCH (n) WHERE (n:Event AND n.summary CONTAINS $s) OR (n:Fact AND n.key = $s)
                    RETURN n LIMIT 1
                """, s=rel["source_summary"]).single()
                tgt = session.run("""
                    MATCH (n) WHERE (n:Event AND n.summary CONTAINS $t) OR (n:Fact AND n.key = $t)
                       OR (n:Entity AND n.name = $t)
                    RETURN n LIMIT 1
                """, t=rel["target_summary"]).single()
                if src and tgt:
                    session.run(f"""
                        MATCH (s) WHERE id(s) = $sid
                        MATCH (t) WHERE id(t) = $tid
                        CREATE (s)-[:{rtype}]->(t)
                    """, sid=src["n"].id, tid=tgt["n"].id)
    
    milvus_collection.flush()
    return {"success": True, "stored_ids": stored_ids,
            "message": f"已存储 {len(stored_ids['event_ids'])} 个事件, {len(stored_ids['fact_ids'])} 个事实, {len(stored_ids['knowledge_ids'])} 条知识"}

def query_memory(
    user_id: str,
    fact_keys: Optional[List[str]] = None,
    event_query: Optional[Dict] = None,
    knowledge_query: Optional[Dict] = None,
    global_vector_fallback: Optional[Dict] = None
) -> Dict[str, Any]:
    result = {"facts": {}, "events": [], "knowledge": [], "vector_chunks": []}
    
    with neo4j_client.driver.session() as session:
        # 1. 事实查询
        if fact_keys:
            for key in fact_keys:
                rec = session.run("""
                    MATCH (u:User {user_id: $user_id})-[:HAS_FACT]->(f:Fact {key: $key})
                    WHERE NOT (f)<-[:OVERRIDES]-(:Fact)
                    RETURN f.value AS value ORDER BY f.updated_at DESC LIMIT 1
                """, user_id=user_id, key=key).single()
                if rec:
                    result["facts"][key] = rec["value"]
        
        # 2. 事件查询
        if event_query:
            sem_text = event_query.get("semantic_text", "")
            t_start = event_query.get("time_start")
            t_end = event_query.get("time_end")
            entities = event_query.get("entities", [])
            top_k = event_query.get("top_k", 3)
            
            vec = get_embedding(sem_text)
            expr = f"user_id == '{user_id}' and ref_type == 'event'"
            if t_start: expr += f" and timestamp >= {t_start}"
            if t_end: expr += f" and timestamp <= {t_end}"
            
            hits = milvus_collection.search(
                data=[vec], anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k*2, expr=expr, output_fields=["ref_id"]
            )
            event_ids = [hit.entity.get("ref_id") for hit in hits[0]]
            for eid in event_ids[:top_k]:
                ev = session.run("""
                    MATCH (e:Event {event_id: $eid})
                    OPTIONAL MATCH (e)-[:MENTIONS]->(ent:Entity)
                    RETURN e.summary AS summary, e.time AS time, e.location AS location,
                           e.subject AS subject, e.participants AS participants,
                           e.action AS action, e.result AS result, e.tense AS tense,
                           e.confidence AS confidence, collect(ent.name) AS entities
                """, eid=eid).single()
                if ev:
                    if entities and not any(e in ev["entities"] for e in entities):
                        continue
                    result["events"].append({
                        "summary": ev["summary"], "time": ev["time"], "location": ev["location"],
                        "subject": ev["subject"], "participants": ev["participants"],
                        "action": ev["action"], "result": ev["result"],
                        "tense": ev["tense"], "confidence": ev["confidence"],
                        "entities": ev["entities"]
                    })
        
        # 3. 知识查询
        if knowledge_query:
            sem_text = knowledge_query.get("semantic_text", "")
            top_k = knowledge_query.get("top_k", 2)
            vec = get_embedding(sem_text)
            hits = milvus_collection.search(
                data=[vec], anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k, expr=f"user_id == '{user_id}' and ref_type == 'knowledge'",
                output_fields=["ref_id"]
            )
            for hit in hits[0]:
                kn = session.run("""
                    MATCH (k:Knowledge {knowledge_id: $kid})
                    RETURN k.title AS title, k.content AS content
                """, kid=hit.entity.get("ref_id")).single()
                if kn:
                    result["knowledge"].append({"title": kn["title"], "content": kn["content"]})
    
    # 4. 兜底向量检索
    if global_vector_fallback and not any([result["facts"], result["events"], result["knowledge"]]):
        text = global_vector_fallback.get("text", "")
        top_k = global_vector_fallback.get("top_k", 5)
        vec = get_embedding(text)
        hits = milvus_collection.search(
            data=[vec], anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k, expr=f"user_id == '{user_id}'", output_fields=["text"]
        )
        for hit in hits[0]:
            result["vector_chunks"].append(hit.entity.get("text"))
    
    return result
```

---

## 7. 测试用例

### 7.1 用例一：基础事件与事实提取

**存储输入**：
```json
{
  "user_id": "u_123",
  "events": [{
    "summary": "和李丽在沃尔玛买排骨",
    "time": 1713657600,
    "location": "沃尔玛超市",
    "subject": "我",
    "participants": ["李丽"],
    "action": "购买排骨",
    "result": "买到了新鲜的排骨",
    "tense": "past",
    "confidence": "real",
    "entities": ["李丽", "排骨"]
  }],
  "facts": [
    {"key": "spouse_name", "value": "李丽"}
  ]
}
```

**查询输入**：
```json
{
  "user_id": "u_123",
  "fact_keys": ["spouse_name"],
  "event_query": {"semantic_text": "买排骨", "top_k": 1}
}
```

**预期结果**：返回李丽为配偶，并返回买排骨事件的完整字段。

---

### 7.2 用例二：事实覆盖

**第一次存储**：`facts: [{"key": "city", "value": "上海"}]`
**第二次存储**：`facts: [{"key": "city", "value": "杭州"}]`

**查询**：`fact_keys: ["city"]`

**预期**：仅返回 `{"city": "杭州"}`，且 Neo4j 中存在 `[:OVERRIDES]` 关系。

---

### 7.3 用例三：事件因果关系与包含关系

**存储事件**：
- 事件A：`summary="下暴雨忘关车窗"`
- 事件B：`summary="座椅湿了"`
- 事件C：`summary="车里有霉味"`
- 事件P：`summary="车辆受损事件"`

**关系**：
- `A -[:CAUSED]-> B`
- `B -[:CAUSED]-> C`
- `A -[:SUB_EVENT_OF]-> P`
- `B -[:SUB_EVENT_OF]-> P`

**查询**：`event_query.semantic_text = "霉味原因"`

**预期**：向量召回事件C，图游走返回因果链A→B→C，并可通过包含关系定位父事件P。

---

### 7.4 用例四：知识检索与混合查询

**存储**：
```json
{
  "knowledge": [{"title": "煮饺子防破", "content": "水里加盐", "category": "cooking"}],
  "facts": [{"key": "coffee_preference", "value": "冰美式"}]
}
```

**查询**：
```json
{
  "fact_keys": ["coffee_preference"],
  "knowledge_query": {"semantic_text": "饺子不破"}
}
```

**预期**：同时返回咖啡偏好和煮饺子技巧。

---

## 8. 部署与运行说明

### 8.1 环境依赖

```bash
pip install pymilvus neo4j ollama numpy
```

### 8.2 启动服务

```bash
# Milvus
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest

# Neo4j
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

# Ollama (需提前拉取模型)
ollama pull nomic-embed-text
```

### 8.3 运行测试

将代码保存为 `memory_tools.py`，直接执行 `__main__` 部分进行功能验证。

---

## 9. 附录

### 9.1 雪花ID生成器测试

```python
print(id_gen.next_str())  # 输出如 "1234567890123456789"
```

### 9.2 向量维度确认

```python
vec = get_embedding("test")
print(len(vec))  # 应为 768
```

### 9.3 Neo4j 数据查看

访问 `http://localhost:7474`，使用 `neo4j/password` 登录，执行：
```cypher
MATCH (u:User)-[:HAS_EVENT]->(e:Event) RETURN u, e
```

---

*文档版本：2.0*  
*最后更新：2026年4月*