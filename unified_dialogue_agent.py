"""
统一对话：JSON ReAct 循环（与 schema_react_agent 同协议），组合 store_memory / query_memory / fetch_older_chat。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import ollama

from config import settings
from dialogue_store import DialogueStore
from memory_service import MemorySystem
from permanent_memory_store import PermanentMemoryStore
from schema_react_agent import _parse_react_json

log = logging.getLogger(__name__)

MAX_FETCH_OLDER_CHAT_CALLS = 8


_UNIFIED_REACT_HEAD = (
    """你是「统一对话与长期记忆」助手，通过多步工具调用回答用户本轮问题。

## 硬性规则（必须遵守）
1) **store_memory**：只能根据「本轮用户输入」中**明确出现的新信息**抽取记忆项；不得把上方「历史对话」里的内容当作新事实写入记忆（历史仅用于理解指代与语境）。
2) **query_memory**：用于检索用户既有长期记忆，辅助回答。
3) **fetch_older_chat**：当最近对话窗口信息不足、且需要更早的聊天记录时使用；参数 **before_id** 须为已见过的消息 **id**（整数），**limit** 不超过 """
    + str(settings.dialogue_older_fetch_max)
    + """。
4) 工具名必须完全匹配下文列表之一；结束时必须使用 **final_answer**。
5) **ReAct 步数**：下文「ReAct 步数限制」会标明本轮**最多步数**与**当前步数**；请在用尽步数前给出 **final_answer**。

## 每轮输出格式（仅输出一个 JSON 对象，不要 Markdown 围栏）
{
  "thought": "本步推理（中文）",
  "tool": "<工具名>",
  "arguments": { ... }
}

## 工具名与 arguments
"""
)

_UNIFIED_REACT_TOOLS_1_3 = """1) store_memory — 写入长期记忆（memories / relations 结构与后端 MemorySystem 一致）
   {"memories": [...], "relations": [] 或省略}
2) query_memory — 语义检索
   {"query_text": "...", "memory_types": ["event","fact","knowledge"] 或省略, "top_k": 5 或省略,
    "time_start": 可选整数时间戳, "time_end": 可选整数时间戳, "tense": 可选, "confidence": 可选}
3) fetch_older_chat — 读取比 before_id 更早的对话行（服务端参数化只读 SQL）
   {"before_id": 整数, "limit": 可选整数}
"""

_UNIFIED_REACT_TOOL_UPDATE_PM = """4) update_permanent_memory — 更新永驻记忆（写入 MySQL；类别须为下列中文之一）
   {"category": "用户身份" 或 "智能体性格" 或 "工作规范", "value": "字符串，最多1000字"}
"""

_UNIFIED_REACT_FINAL_4 = """4) final_answer — 结束并给用户完整中文回答
   {"answer": "..."}
"""

_UNIFIED_REACT_FINAL_5 = """5) final_answer — 结束并给用户完整中文回答
   {"answer": "..."}
"""


def _build_unified_system(*, permanent_block: str, include_pm_tool: bool) -> str:
    tail = (
        _UNIFIED_REACT_TOOLS_1_3 + _UNIFIED_REACT_TOOL_UPDATE_PM + _UNIFIED_REACT_FINAL_5
        if include_pm_tool
        else _UNIFIED_REACT_TOOLS_1_3 + _UNIFIED_REACT_FINAL_4
    )
    body = _UNIFIED_REACT_HEAD + tail
    pb = (permanent_block or "").strip()
    if pb:
        return pb + "\n\n" + body
    return body


# 默认无永驻工具（与旧行为一致，便于测试直接引用）
UNIFIED_REACT_SYSTEM = _build_unified_system(permanent_block="", include_pm_tool=False)


def _default_llm_chat(messages: List[Dict[str, Any]]) -> str:
    client = ollama.Client(
        host=settings.ollama_base_url,
        timeout=settings.ollama_request_timeout_s,
    )
    resp = client.chat(
        model=settings.ollama_chat_model,
        messages=messages,
    )
    return (resp.get("message") or {}).get("content") or ""


def _format_history_block(recent_rows: List[Dict[str, Any]]) -> str:
    if not recent_rows:
        return "（无历史记录）"
    lines: List[str] = []
    for r in recent_rows:
        rid = r.get("id")
        role = (r.get("role") or "").strip()
        body = (r.get("content") or "").strip()
        lines.append(f"id={rid} [{role}] {body}")
    return "\n".join(lines)


def _last_react_round_excerpt_for_user(trace: List[Dict[str, Any]]) -> str:
    """步数用尽且未 final_answer 时，拼给用户看的最后一轮思考与工具信息。"""
    if not trace:
        return ""
    last = trace[-1]
    if "thought" in last and "tool" in last:
        args = last.get("arguments")
        if not isinstance(args, dict):
            args = {}
        try:
            args_s = json.dumps(args, ensure_ascii=False)
        except TypeError:
            args_s = str(args)
        return (
            "【最后一轮模型输出】\n"
            f"- 思考：{last.get('thought', '')}\n"
            f"- 工具：{last.get('tool', '')}\n"
            f"- 参数：{args_s}"
        )
    if last.get("error"):
        raw = (last.get("raw_preview") or "").strip()
        return (
            "【最后一轮】模型输出解析失败\n"
            f"- 原因：{last.get('error', '')}\n"
            + (f"- 原始片段：{raw}" if raw else "")
        )
    return ""


def run_unified_dialogue(
    mem: MemorySystem,
    dialogue: DialogueStore,
    user_message: str,
    recent_rows: List[Dict[str, Any]],
    *,
    user_id: Optional[str] = None,
    max_steps: Optional[int] = None,
    llm_chat: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    perm_store: Optional[PermanentMemoryStore] = None,
) -> Dict[str, Any]:
    """
    ReAct 主循环：返回 final_answer、trace；内部直接调用 MemorySystem 与 DialogueStore。
    """
    uid = (user_id or settings.default_user_id or "").strip() or "default"
    llm_fn = llm_chat or _default_llm_chat
    cap = max(1, min(int(settings.unified_dialogue_max_steps_cap), 40))
    max_steps = max_steps if max_steps is not None else settings.unified_dialogue_max_steps
    max_steps = max(1, min(int(max_steps), cap))
    trace: List[Dict[str, Any]] = []
    older_calls = 0

    history_block = _format_history_block(recent_rows)
    perm_block = perm_store.format_prompt_block(uid) if perm_store else ""
    system_base = _build_unified_system(
        permanent_block=perm_block,
        include_pm_tool=perm_store is not None,
    )

    def _system_with_step_progress(step_idx: int) -> str:
        return (
            system_base
            + "\n\n## ReAct 步数限制\n"
            + f"- **本轮最多 {max_steps} 步**（每步输出一个 JSON：thought、tool、arguments；含工具调用与 final_answer）。\n"
            + f"- **当前为第 {step_idx + 1} 步**（从 1 计数）。\n"
            + "- 若已能完整回答用户，请直接输出 **final_answer**，避免在接近上限时仍只调工具而不结束。"
        )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _system_with_step_progress(0)},
        {
            "role": "user",
            "content": (
                "## 最近对话（MySQL，不含本轮；含 id 供 fetch_older_chat 游标）\n"
                f"{history_block}\n\n"
                "## 本轮用户输入\n"
                f"{user_message}\n\n"
                f"当前用户分区 user_id={uid!r}（store_memory / query_memory 必须使用此维度）。\n"
                "请开始：需要时调用工具；能回答时使用 final_answer。"
            ),
        },
    ]

    final_answer: Optional[str] = None

    for step in range(max_steps):
        messages[0]["content"] = _system_with_step_progress(step)
        raw = ""
        try:
            raw = llm_fn(messages) or ""
            obj = _parse_react_json(raw)
        except Exception as e:
            trace.append(
                {
                    "step": step + 1,
                    "error": f"解析或模型输出失败: {e}",
                    "raw_preview": raw[:400] if raw else "",
                }
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": "上一步输出不是合法 JSON。请严格输出一个 JSON 对象，字段为 thought, tool, arguments。",
                }
            )
            continue

        thought = obj.get("thought", "")
        tool = (obj.get("tool") or "").strip()
        args = obj.get("arguments")
        if not isinstance(args, dict):
            args = {}

        if tool == "final_answer":
            final_answer = str(args.get("answer", "")).strip() or "（空回答）"
            trace.append(
                {
                    "step": step + 1,
                    "thought": thought,
                    "tool": tool,
                    "arguments": args,
                    "observation": {"done": True},
                }
            )
            break

        observation: Dict[str, Any]
        if tool == "store_memory":
            memories = args.get("memories")
            relations = args.get("relations")
            if not isinstance(memories, list):
                memories = []
            if relations is not None and not isinstance(relations, list):
                observation = {"error": "relations 须为数组或省略"}
            else:
                rel = relations if isinstance(relations, list) else []
                try:
                    observation = mem.store_memory(uid, memories=memories, relations=rel)
                except Exception as e:
                    observation = {"error": f"store_memory 失败: {e}"}
        elif tool == "query_memory":
            qt = str(args.get("query_text") or "").strip()
            mtypes = args.get("memory_types")
            if mtypes is not None and not isinstance(mtypes, list):
                observation = {"error": "memory_types 须为字符串数组或省略"}
            else:
                top_k = args.get("top_k", 5)
                try:
                    top_k_i = int(top_k)
                except (TypeError, ValueError):
                    top_k_i = 5
                ts = args.get("time_start")
                te = args.get("time_end")
                time_start = int(ts) if ts is not None and str(ts).strip() != "" else None
                time_end = int(te) if te is not None and str(te).strip() != "" else None
                tense = args.get("tense")
                confidence = args.get("confidence")
                observation = mem.query_memory(
                    uid,
                    qt,
                    memory_types=mtypes if isinstance(mtypes, list) else None,
                    time_start=time_start,
                    time_end=time_end,
                    top_k=max(1, min(top_k_i, 50)),
                    tense=str(tense).strip() if tense else None,
                    confidence=str(confidence).strip() if confidence else None,
                )
        elif tool == "fetch_older_chat":
            if older_calls >= MAX_FETCH_OLDER_CHAT_CALLS:
                observation = {
                    "error": (
                        f"fetch_older_chat 调用次数已达上限 {MAX_FETCH_OLDER_CHAT_CALLS}，"
                        "请使用 final_answer 总结或缩小范围。"
                    ),
                }
            else:
                older_calls += 1
                bid = args.get("before_id")
                lim = args.get("limit", settings.dialogue_older_fetch_max)
                try:
                    before_id = int(bid)
                except (TypeError, ValueError):
                    observation = {"error": "fetch_older_chat 需要有效的整数 arguments.before_id"}
                else:
                    try:
                        lim_i = int(lim)
                    except (TypeError, ValueError):
                        lim_i = settings.dialogue_older_fetch_max
                    try:
                        rows = dialogue.fetch_older(uid, before_id, lim_i)
                        observation = {"rows": rows, "count": len(rows)}
                    except Exception as e:
                        observation = {"error": f"fetch_older_chat 失败: {e}"}
        elif tool == "update_permanent_memory":
            if perm_store is None:
                observation = {"error": "永驻记忆服务不可用"}
            else:
                cat = args.get("category")
                val = args.get("value", "")
                if not isinstance(val, str):
                    val = str(val) if val is not None else ""
                observation = perm_store.upsert(uid, str(cat or ""), val)
        else:
            allowed = "store_memory, query_memory, fetch_older_chat, final_answer"
            if perm_store is not None:
                allowed += ", update_permanent_memory"
            observation = {"error": f"未知工具: {tool!r}，允许: {allowed}"}

        log.info(
            "unified ReAct step=%s tool=%s args_keys=%s",
            step + 1,
            tool,
            list(args.keys()),
        )
        trace.append(
            {
                "step": step + 1,
                "thought": thought,
                "tool": tool,
                "arguments": args,
                "observation": observation,
            }
        )

        messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
        messages.append(
            {
                "role": "user",
                "content": "观察结果（JSON，勿重复无意义的相同调用）：\n"
                + json.dumps(observation, ensure_ascii=False, default=str),
            }
        )

    if final_answer is None:
        base = (
            f"（未在 {max_steps} 步内给出 final_answer；可适当提高 max_steps（上限 {cap}）或简化问题。）"
        )
        suffix = _last_react_round_excerpt_for_user(trace)
        final_answer = base + (f"\n\n{suffix}" if suffix else "")

    return {
        "agent_mode": "unified_react",
        "final_answer": final_answer,
        "trace": trace,
        "fetch_older_chat_calls": older_calls,
        "steps_used": len(trace),
    }
