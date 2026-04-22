from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 对话模型（编排 store/query）；嵌入模型须与 vector_dim 一致，均可用 .env 覆盖
    ollama_chat_model: str = "qwen3.5:35b-a3b-q8_0"
    ollama_embed_model: str = "qwen3-embedding:8b"
    # Qwen3 嵌入支持 MRL：若设置则须与 vector_dim 一致；不设则使用模型默认满维（8B 多为 4096）
    ollama_embed_dimensions: Optional[int] = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_request_timeout_s: float = 120.0
    ollama_embed_retries: int = 3

    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_collection: str = "memory_vectors"
    # 须与当前嵌入模型输出维度一致（qwen3-embedding:8b 默认多为 4096）
    vector_dim: int = 4096
    milvus_search_ef: int = 64

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "admin123"

    snowflake_datacenter_id: int = 1
    snowflake_worker_id: int = 1

    # 自然语言「单参数」接口使用的默认用户维度（Milvus/Neo4j 分区）
    default_user_id: str = "default"

    # 结构化查询助手：目标 MySQL（须使用只读账号），如 mysql+pymysql://user:pass@host:3306/db
    mysql_url: str = "mysql+pymysql://ai_agent:readonly@localhost:3306/agent_schema_demo"

    # 统一对话 ReAct：对话消息表（须 SELECT + INSERT；可与 mysql_url 同库但账号需写权限）
    # 空字符串表示沿用 mysql_url（请确保该账号具备 INSERT 权限）
    mysql_dialogue_url: str = ""
    dialogue_table: str = "chat_messages"
    dialogue_col_id: str = "id"
    dialogue_col_user_id: str = "user_id"
    dialogue_col_role: str = "role"
    dialogue_col_content: str = "content"
    dialogue_col_created_at: str = "created_at"
    dialogue_fetch_limit: int = 10
    dialogue_older_fetch_max: int = 20
    # 统一对话 ReAct：默认步数与硬上限（含工具步与 final_answer）
    unified_dialogue_max_steps: int = 5
    unified_dialogue_max_steps_cap: int = 8

    # 永驻记忆（与对话库同一 MySQL 连接策略；须 SELECT + INSERT/UPDATE）
    permanent_memory_table: str = "permanent_memory"


settings = Settings()
