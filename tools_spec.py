"""OpenAI 风格 function tools，供 Ollama 对话编排（memory_agent）注入。"""
from typing import Any, Dict

STORE_MEMORY_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "store_memory",
        "description": "将用户对话中提取的事件、事实、知识以及它们之间的关系写入长期记忆数据库。事件采用标准8字段结构。",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户唯一标识"},
                "events": {
                    "type": "array",
                    "description": "提取的事件列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "time": {"type": "integer"},
                            "location": {"type": "string"},
                            "subject": {"type": "string"},
                            "participants": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "action": {"type": "string"},
                            "result": {"type": "string"},
                            "tense": {
                                "type": "string",
                                "enum": ["past", "present", "future"],
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["real", "imagined", "planned"],
                            },
                            "entities": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "temp_id": {
                                "type": "string",
                                "description": "同次 store_memory 内唯一临时 id；relations 中可用 source_summary/target_summary 填此 id 以精确绑定节点",
                            },
                        },
                        "required": ["summary", "action"],
                    },
                },
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string"},
                            "value": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "temp_id": {
                                "type": "string",
                                "description": "同次 store_memory 内唯一临时 id，供 relations 引用",
                            },
                        },
                        "required": ["key", "value"],
                    },
                },
                "knowledge": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "category": {"type": "string"},
                            "temp_id": {
                                "type": "string",
                                "description": "同次 store_memory 内唯一临时 id，供 relations 引用",
                            },
                        },
                        "required": ["title", "content"],
                    },
                },
                "relations": {
                    "type": "array",
                    "description": "关系边。SUB_EVENT_OF：source=子事件、target=父事件。RELATED_TO：source=事件、target=知识（target_type=knowledge）。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": [
                                    "NEXT",
                                    "CAUSED",
                                    "SUB_EVENT_OF",
                                    "MENTIONS",
                                    "OVERRIDES",
                                    "RELATED_TO",
                                ],
                            },
                            "source_type": {
                                "type": "string",
                                "enum": ["event", "fact"],
                                "description": "RELATED_TO 时须为 event",
                            },
                            "source_summary": {
                                "type": "string",
                                "description": "事件/事实摘要、英文键名、实体名，或同批次的 temp_id",
                            },
                            "target_type": {
                                "type": "string",
                                "enum": ["event", "fact", "entity", "knowledge"],
                            },
                            "target_summary": {
                                "type": "string",
                                "description": "目标节点匹配串；knowledge 可为标题精确匹配或 kn_ 开头的 knowledge_id；可为 temp_id",
                            },
                        },
                        "required": [
                            "type",
                            "source_type",
                            "source_summary",
                            "target_type",
                            "target_summary",
                        ],
                    },
                },
            },
            "required": ["user_id"],
        },
    },
}

QUERY_MEMORY_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "query_memory",
        "description": "从用户长期记忆中查询事件、事实或知识。支持混合查询，返回结构化结果和语义片段。",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "fact_keys": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "event_query": {
                    "type": "object",
                    "properties": {
                        "semantic_text": {"type": "string"},
                        "time_start": {"type": "integer"},
                        "time_end": {"type": "integer"},
                        "entities": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer", "default": 3},
                    },
                },
                "knowledge_query": {
                    "type": "object",
                    "properties": {
                        "semantic_text": {"type": "string"},
                        "category": {"type": "string"},
                        "top_k": {"type": "integer", "default": 2},
                    },
                },
                "global_vector_fallback": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                },
            },
            "required": ["user_id"],
        },
    },
}
