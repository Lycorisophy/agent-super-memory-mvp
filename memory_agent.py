"""
通过本地 Ollama 对话：系统提示词 + 工具定义 + 用户一句输入 → 工具执行 → 模型自然语言回复。
提示词与工具 schema 对齐 design.md v4 与 v5.0 增量（tense / confidence）。
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

STORE_SYSTEM_TEMPLATE = """你是长期记忆编码助手。请将用户输入中值得保留的信息提取为记忆条目，调用 `store_memory` 工具存储。

每条记忆必须遵循以下统一模板（使用中文全角方括号标注字段）：
【类型】{{事件|事实|知识}}
【时间】{{YYYY-MM-DD HH:MM，不明确则用当前时间}}
【地点】{{地点，无则留空}}
【主体】{{主要关联人，无则留空}}
【内容】{{核心描述。事件：动作+结果；事实：键=值；知识：标题+要点}}
【来源】{{原始对话片段或推断依据}}

提取原则：
- 不要编造信息，缺失字段可留空。
- 事件之间的顺序、因果、包含关系请在 `relations` 中声明，通过临时ID引用（如索引0表示第一条记忆，或使用每条 memories 的 `temp_id`）。
- 事实覆盖关系由系统自动处理，无需显式声明 OVERRIDES。
- **（v5）时态与置信度**：每条记忆可填 `tense`（`past` / `present` / `future`）与 `confidence`（`real` / `imagined` / `planned`）；事件尤应填写。梦见、虚构、梦境类用 `confidence=imagined` 并选对 `tense`；未来计划用 `confidence=planned` 且常配合 `tense=future`。

调用约束：
- user_id 必须为：{user_id}
- 当前时间：{current_time}

工具执行成功后，用中文简短确认已记住的内容。
"""


QUERY_SYSTEM_TEMPLATE = """你是记忆查询路由专家。根据用户问题构造 `query_memory` 调用参数。

参数说明：
- query_text：从用户问题中提炼的语义检索文本。
- memory_types：根据问题类型限定，如问「做了什么」用 ["event"]，问「我的XX是什么」用 ["fact"]。
- 若有时间限定（如「上周」），估算并填写 time_start 和 time_end（Unix秒）。
- **（v5）结构化过滤**：若用户明确限定经历性质（如「过去的经历」「计划中的事」「梦里的事」），可设置 `tense` 或 `confidence` 与 `query_memory` 参数对应，以便精确筛选。

调用约束：
- user_id 必须为：{user_id}
- 当前时间：{current_time}

得到结果后，用自然语言回答用户，不要输出 JSON。
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
    args["user_id"] = settings.default_user_id
    log.info(
        "%s 执行工具 %s user_id=%s arg_top_keys=%s",
        trace,
        name,
        args["user_id"],
        list(args.keys())[:12],
    )
    if name == "query_memory":
        log.debug("%s query_memory 参数 keys=%s", trace, list(args.keys()))
    try:
        if name == "store_memory":
            return mem.store_memory(
                user_id=args["user_id"],
                memories=args.get("memories"),
                relations=args.get("relations"),
                events=args.get("events"),
                facts=args.get("facts"),
                knowledge=args.get("knowledge"),
            )
        if name == "query_memory":
            qt = (args.get("query_text") or "").strip()
            if not qt:
                parts: List[str] = []
                gvf = args.get("global_vector_fallback")
                if isinstance(gvf, dict):
                    t = (gvf.get("text") or "").strip()
                    if t:
                        parts.append(t)
                fk = args.get("fact_keys")
                if isinstance(fk, list) and fk:
                    parts.append(" ".join(str(x) for x in fk if str(x).strip()))
                eq = args.get("event_query")
                if isinstance(eq, dict):
                    st = (eq.get("semantic_text") or "").strip()
                    if st:
                        parts.append(st)
                kq = args.get("knowledge_query")
                if isinstance(kq, dict):
                    st = (kq.get("semantic_text") or "").strip()
                    if st:
                        parts.append(st)
                qt = " ".join(parts).strip()
            if not qt:
                return {
                    "error": "缺少 query_text（或可提供 global_vector_fallback / fact_keys 等旧字段以自动拼接）",
                    "memories": [],
                }
            return mem.query_memory(
                user_id=args["user_id"],
                query_text=qt,
                memory_types=args.get("memory_types"),
                time_start=args.get("time_start"),
                time_end=args.get("time_end"),
                top_k=int(args.get("top_k", 5)),
                tense=args.get("tense"),
                confidence=args.get("confidence"),
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
    ct = datetime.now().strftime("%Y-%m-%d %H:%M")
    system = STORE_SYSTEM_TEMPLATE.format(user_id=uid, current_time=ct)
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
    ct = datetime.now().strftime("%Y-%m-%d %H:%M")
    system = QUERY_SYSTEM_TEMPLATE.format(user_id=uid, current_time=ct)
    return _run_loop(
        mem,
        user_input,
        tools=[QUERY_MEMORY_TOOL],
        system=system,
        allowed={"query_memory"},
        trace="[query]",
    )
