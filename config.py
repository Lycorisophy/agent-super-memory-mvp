from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 对话模型（供后续 Agent/编排使用；记忆写入/检索当前仅用嵌入）
    ollama_chat_model: str = "qwen3.5:35b-a3b-q8_0"
    ollama_embed_model: str = "qwen3-embedding:8b"
    # Qwen3 嵌入支持 MRL：若设置则须与 vector_dim 一致；不设则使用模型默认满维（8B 多为 4096）
    ollama_embed_dimensions: Optional[int] = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_request_timeout_s: float = 120.0
    ollama_embed_retries: int = 3

    milvus_host: str = "localhost"
    milvus_port: str = "19530"
    milvus_collection: str = "memory_chunks"
    # qwen3-embedding:8b 默认向量维 4096；若改 ollama_embed_dimensions 请同步修改此值
    vector_dim: int = 4096
    milvus_search_ef: int = 64

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "admin123"

    snowflake_datacenter_id: int = 1
    snowflake_worker_id: int = 1

    # 自然语言「单参数」接口使用的默认用户维度（Milvus/Neo4j 分区）
    default_user_id: str = "default"


settings = Settings()
