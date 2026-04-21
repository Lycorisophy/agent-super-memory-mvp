"""
通过本地 Ollama 对话：系统提示词 + 工具定义 + 用户一句输入 → 工具执行 → 模型自然语言回复。
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Set

import ollama

from config import settings
from memory_service import MemorySystem
from tools_spec import QUERY_MEMORY_TOOL, STORE_MEMORY_TOOL

log = logging.getLogger(__name__)

STORE_SYSTEM_TEMPLATE = """你是长期记忆编码助手。请分析「用户输入」中值得持久化的信息，并调用 `store_memory` 工具写入数据库。若无值得存储的内容，可不调用工具。

### 提取规范

**1. 事件**
- 必填字段：`summary`（一句话概括）、`action`（具体行为）。
- 其他字段尽量补全，若用户未提及则留空，严禁凭空编造：
  - `time`：发生时间的Unix秒数，无明确时间则用当前时间。
  - `location`：地点。
  - `subject`：动作主体（通常为“我”或用户名）。
  - `participants`：参与人员名单。
  - `result`：事件结果或造成的影响。
  - `tense`：时态，可选 `past`（过去）、`present`（现在）、`future`（将来）。
  - `confidence`：置信度，可选 `real`（真实发生）、`imagined`（想象/虚构）、`planned`（计划中）。
  - `entities`：事件中涉及的人名、地名、机构名列表。

**2. 事实**
- 指稳定的属性或状态，采用标准化键名（推荐英文，如 `spouse_name`、`city`）。
- 若同一事实多次出现，系统会自动管理版本，你只需提取当前正确的值。

**3. 知识**
- 客观方法、规律或外部信息，包含 `title`（标题）和 `content`（内容），可加 `category`（类别）。

**4. 关系**
- 仅在明确存在以下逻辑时填写：
  - `NEXT`：事件 A 紧接着事件 B 发生。
  - `CAUSED`：事件 A 导致了事件 B。
  - `SUB_EVENT_OF`：子事件指向父事件（注意方向：源头是子事件，目标是父事件）。
  - `MENTIONS`：事件提及了某个实体。
- 事实的版本覆盖关系由系统自动处理，无需填写。

### 调用须知
- 调用工具时，参数 `user_id` 必须且只能填写：{user_id}
- 当前时间戳（秒）：{current_ts}

### 回复要求
工具执行成功后，用一两句中文向用户确认已记住的内容，不要展示 JSON 或工具原始返回。
"""


QUERY_SYSTEM_TEMPLATE = """你是记忆查询路由专家。根据「用户输入」判断是否需要检索长期记忆。

- **需要调用 `query_memory`**：问题涉及用户个人信息、过往经历、个人偏好、已存储的知识。
- **无需调用工具**：纯寒暄或与记忆无关的问题，直接自然语言回复。

### 查询参数构造指南

**1. 事实查询（fact_keys）**
- 根据用户问题，列出可能的标准化键名（英文优先，如 `spouse_name`、`city`、`food_preference`）。
- **注意**：`fact_keys` 仅用于辅助向量检索，系统会进行语义匹配而非精确字符串比对，因此提供多个候选键名有助于提高召回率。
- 必须同时提供 `global_vector_fallback.text`，内容为用户原话。

**2. 事件回忆（event_query）**
- `semantic_text`：描述用户想查找的事件内容（如“买排骨”“去杭州出差”）。
- 若有时间限定（如“上周”“上个月”），估算并填写 `time_start` 和 `time_end`（Unix 秒）。
- 若涉及特定人物或地点，将名称填入 `entities` 列表以提高精度。

**3. 知识检索（knowledge_query）**
- `semantic_text`：描述用户想查找的知识或方法（如“饺子怎么煮不破”）。

**4. 兜底检索（global_vector_fallback）**
- 务必设置 `global_vector_fallback.text` 为用户原话，当结构化查询无结果时用于全库语义召回。

### 调用约束
- 调用 `query_memory` 时，参数 `user_id` 必须且只能使用：{user_id}
- 当前时间参考：{current_human}

### 回复要求
获得工具结果后，仅用自然语言回答用户，不得输出 JSON 或工具原始返回内容。
"""


def _normalize_tool_arguments(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        return json.loads(raw)
    return {}


def _assistant_message_dict(message: Any) -> Dict[str, Any]:
    if hasattr(message, "model_dump"):
        d = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        d = dict(message)
    else:
        d = {"role": getattr(message, "role", "assistant")}
        if getattr(message, "content", None):
            d["content"] = message.content
        if getattr(message, "tool_calls", None):
            d["tool_calls"] = message.tool_calls

    d["role"] = "assistant"
    tcs = d.get("tool_calls")
    if not tcs:
        return d

    fixed: List[Dict[str, Any]] = []
    for tc in tcs:
        if isinstance(tc, dict):
            fn = dict(tc.get("function") or {})
            args = fn.get("arguments")
        else:
            fn_obj = tc.function
            fn = {"name": fn_obj.name, "arguments": fn_obj.arguments}
            args = fn["arguments"]
        # Ollama Python 客户端要求 tool_calls[].function.arguments 为 dict，不能是 JSON 字符串
        fn["arguments"] = _normalize_tool_arguments(args)
        fixed.append({"function": fn})
    d["tool_calls"] = fixed
    return d


def _tool_call_iter(message: Any) -> List[tuple[str, Dict[str, Any]]]:
    out: List[tuple[str, Dict[str, Any]]] = []
    tcs = getattr(message, "tool_calls", None) or []
    for tc in tcs:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            args = _normalize_tool_arguments(fn.get("arguments"))
        else:
            name = tc.function.name
            args = _normalize_tool_arguments(tc.function.arguments)
        if name:
            out.append((name, args))
    return out


def _run_tool(
    mem: MemorySystem, name: str, args: Dict[str, Any], trace: str
) -> Dict[str, Any]:
    args = dict(args)
    # 自然语言单参数接口：用户身份仅由服务端 default_user_id 决定
    args["user_id"] = settings.default_user_id
    log.info(
        "%s 执行工具 %s user_id=%s arg_top_keys=%s",
        trace,
        name,
        args["user_id"],
        list(args.keys())[:12],
    )
    if name == "query_memory":
        log.debug(
            "%s query_memory 参数详情 fact_keys=%r event_query=%s knowledge_query=%s "
            "global_vector_fallback=%s",
            trace,
            args.get("fact_keys"),
            args.get("event_query"),
            args.get("knowledge_query"),
            args.get("global_vector_fallback"),
        )
    try:
        if name == "store_memory":
            return mem.store_memory(
                user_id=args["user_id"],
                events=args.get("events"),
                facts=args.get("facts"),
                knowledge=args.get("knowledge"),
                relations=args.get("relations"),
            )
        if name == "query_memory":
            return mem.query_memory(
                user_id=args["user_id"],
                fact_keys=args.get("fact_keys"),
                event_query=args.get("event_query"),
                knowledge_query=args.get("knowledge_query"),
                global_vector_fallback=args.get("global_vector_fallback"),
            )
        return {"error": f"未知工具: {name}"}
    except Exception as e:
        log.warning("%s 工具 %s 异常: %s", trace, name, e)
        return {"error": str(e)}


def _run_loop(
    mem: MemorySystem,
    user_input: str,
    tools: List[Dict[str, Any]],
    system: str,
    allowed: Set[str],
    trace: str,
) -> Dict[str, Any]:
    client = ollama.Client(host=settings.ollama_base_url)
    model = settings.ollama_chat_model
    log.info(
        "%s 开始 ollama=%s model=%s user_input_len=%d allowed_tools=%s",
        trace,
        settings.ollama_base_url,
        model,
        len(user_input),
        sorted(allowed),
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]
    tool_results_log: List[Dict[str, Any]] = []
    last_reply = ""

    for round_idx in range(8):
        log.info(
            "%s Ollama chat 第 %d 轮 messages=%d",
            trace,
            round_idx + 1,
            len(messages),
        )
        resp = client.chat(
            model=model,
            messages=messages,
            tools=tools,
            stream=False,
        )
        msg = resp.message
        last_reply = (getattr(msg, "content", None) or "").strip()
        calls = _tool_call_iter(msg)
        log.info(
            "%s 第 %d 轮返回 content_len=%d tool_call_count=%d",
            trace,
            round_idx + 1,
            len(last_reply),
            len(calls),
        )
        if not calls:
            log.info("%s 无工具调用，结束编排", trace)
            break

        messages.append(_assistant_message_dict(msg))
        for name, args in calls:
            if name not in allowed:
                result: Dict[str, Any] = {
                    "error": f"本会话仅允许工具: {sorted(allowed)}",
                }
                log.warning("%s 拒绝工具 %s（不在白名单）", trace, name)
            else:
                result = _run_tool(mem, name, args, trace)
            err = result.get("error") if isinstance(result, dict) else None
            log.info(
                "%s 工具 %s 返回 keys=%s error=%s",
                trace,
                name,
                list(result.keys())[:10] if isinstance(result, dict) else type(result),
                err,
            )
            tool_results_log.append({"tool": name, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    log.info(
        "%s 结束 tool_called=%s tool_invocations=%d reply_len=%d",
        trace,
        bool(tool_results_log),
        len(tool_results_log),
        len(last_reply),
    )
    return {
        "reply": last_reply,
        "tool_called": bool(tool_results_log),
        "tool_results": tool_results_log,
    }


def run_store_assistant(mem: MemorySystem, user_input: str) -> Dict[str, Any]:
    uid = settings.default_user_id
    system = STORE_SYSTEM_TEMPLATE.format(
        user_id=uid,
        current_ts=int(time.time()),
    )
    return _run_loop(
        mem,
        user_input,
        tools=[STORE_MEMORY_TOOL],
        system=system,
        allowed={"store_memory"},
        trace="[store]",
    )


def run_query_assistant(mem: MemorySystem, user_input: str) -> Dict[str, Any]:
    uid = settings.default_user_id
    system = QUERY_SYSTEM_TEMPLATE.format(
        user_id=uid,
        current_human=datetime.now().isoformat(timespec="seconds"),
    )
    return _run_loop(
        mem,
        user_input,
        tools=[QUERY_MEMORY_TOOL],
        system=system,
        allowed={"query_memory"},
        trace="[query]",
    )
