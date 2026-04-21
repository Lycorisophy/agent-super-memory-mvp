"""OpenAI 风格 function tools，与 design.md v4 + v5.0 增量工具定义一致。"""
from typing import Any, Dict

STORE_MEMORY_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "store_memory",
        "description": "将用户对话中提取的记忆条目存储到长期记忆库。每条记忆需遵循统一的中文模板字段；事实版本覆盖由系统自动处理。",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户唯一标识"},
                "memories": {
                    "type": "array",
                    "description": "记忆条目列表；同次调用内可用 temp_id 或数组下标在 relations 中引用",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["event", "fact", "knowledge"],
                                "description": "记忆类型",
                            },
                            "time": {
                                "type": "string",
                                "description": "时间，格式 YYYY-MM-DD HH:MM；可省略则用当前时间",
                            },
                            "location": {"type": "string", "description": "地点"},
                            "subject": {"type": "string", "description": "主体人物"},
                            "content": {
                                "type": "string",
                                "description": "核心内容：事件为动作+结果；事实为「键 = 值」；知识为标题+要点",
                            },
                            "source": {
                                "type": "string",
                                "description": "原始依据或对话片段",
                            },
                            "temp_id": {
                                "type": "string",
                                "description": "同次调用内唯一临时标识，供 relations 引用；省略则可用数组下标",
                            },
                            "tense": {
                                "type": "string",
                                "enum": ["past", "present", "future"],
                                "description": "时态（v5），不明确可不填",
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["real", "imagined", "planned"],
                                "description": "置信度（v5），不明确可不填",
                            },
                        },
                        "required": ["type", "content"],
                    },
                },
                "relations": {
                    "type": "array",
                    "description": "记忆之间的关系；SUB_EVENT_OF：source_temp_id 为子事件、target_temp_id 为父事件。事实 OVERRIDES 由系统自动建立，勿传。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["NEXT", "CAUSED", "SUB_EVENT_OF", "RELATED"],
                            },
                            "source_temp_id": {
                                "type": "string",
                                "description": "源记忆在同批 memories 中的 temp_id 或下标字符串",
                            },
                            "target_temp_id": {
                                "type": "string",
                                "description": "目标记忆的 temp_id 或下标字符串",
                            },
                        },
                        "required": ["type", "source_temp_id", "target_temp_id"],
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
        "description": "从长期记忆中按语义检索相关记忆条目（Milvus + Neo4j）。",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "用户唯一标识"},
                "query_text": {
                    "type": "string",
                    "description": "语义检索文本，从用户问题中提炼",
                },
                "memory_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["event", "fact", "knowledge"],
                    },
                    "description": "限定记忆类型；省略则检索全部类型",
                },
                "time_start": {
                    "type": "integer",
                    "description": "时间范围起点（Unix 秒）",
                },
                "time_end": {
                    "type": "integer",
                    "description": "时间范围终点（Unix 秒）",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回条数上限",
                    "default": 5,
                },
                "tense": {
                    "type": "string",
                    "enum": ["past", "present", "future"],
                    "description": "按事件时态过滤（Neo4j 侧）；未填不过滤",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["real", "imagined", "planned"],
                    "description": "按置信度过滤（Neo4j 侧）；未填不过滤",
                },
            },
            "required": ["user_id", "query_text"],
        },
    },
}
