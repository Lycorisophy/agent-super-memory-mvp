"""
二期：ReAct 风格多步查询（自研循环 + Ollama JSON 协议），工具委托 SchemaManager。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

import ollama

from config import settings
from schema_manager import SchemaManager

log = logging.getLogger(__name__)

MAX_SQL_TOOL_CALLS = 8

REACT_SYSTEM = """你是「结构化数据库查询」助手，通过多步工具调用回答用户问题。

## 当前会话参数
- HTTP 请求里的 mode：dry_run 表示 execute_sql 只校验 SQL、不连 MySQL；auto_execute 表示可真实执行只读 SQL。
- 你必须只生成只读 SQL（SELECT / WITH … / SHOW / DESCRIBE / EXPLAIN）。

## 每轮输出格式（仅输出一个 JSON 对象，不要 Markdown 围栏）
{
  "thought": "本步推理（中文）",
  "tool": "<工具名>",
  "arguments": { ... }
}

## 工具名与 arguments
1) search_tables — 语义检索相关表
   {"query": "自然语言，描述要找的表或业务域"}
2) describe_table — 查看某张表的字段与注释
   {"table": "表名"}
3) execute_sql — 执行一条只读 SQL（dry_run 下仅校验不返回真实行）
   {"sql": "单条 SQL"}
4) get_foreign_keys — 查看 Neo4j 中 RELATES_TO 关系（未导入时为空列表）
   {"table": "可选，表名；省略则返回全部已建模关系"}
5) final_answer — 结束并给用户中文结论
   {"answer": "面向用户的完整回答"}

## 策略
- 先 search_tables / describe_table 弄清结构，再 execute_sql。
- SQL 报错时阅读 observation 中的 error 修正重试（仍受 mode 与次数限制）。
- 能回答时尽快使用 final_answer，不要无限循环。
"""


def _parse_react_json(text: str) -> Dict[str, Any]:
    s = (text or "").strip()
    if not s:
        raise ValueError("模型输出为空")
    if s.startswith("```"):
        s = re.sub(r"^```\w*\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s).strip()
    i = s.find("{")
    j = s.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("输出中未找到 JSON 对象")
    return json.loads(s[i : j + 1])


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


def run_schema_react(
    sm: SchemaManager,
    user_query: str,
    *,
    mode: str = "dry_run",
    min_confidence: Optional[float] = None,
    max_steps: int = 12,
    llm_chat: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
) -> Dict[str, Any]:
    """
    ReAct 主循环：返回 final_answer、trace；不修改单步 query_schema 的响应形状（由路由包装）。
    """
    llm_fn = llm_chat or _default_llm_chat
    max_steps = max(1, min(int(max_steps), 40))
    trace: List[Dict[str, Any]] = []
    sql_tool_calls = 0

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": REACT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"用户问题：{user_query}\n\n"
                f"当前 mode={mode!r}（execute_sql 行为与此一致）。\n"
                f"min_confidence 过滤（传给 search_tables）：{min_confidence!r}\n"
                "请开始：若信息不足先调用工具；否则 final_answer。"
            ),
        },
    ]

    final_answer: Optional[str] = None

    for step in range(max_steps):
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
        if tool == "search_tables":
            q = str(args.get("query") or args.get("q") or "").strip() or user_query
            observation = sm.search_schema_tables(q, min_confidence=min_confidence, limit=8)
        elif tool == "describe_table":
            tname = str(args.get("table") or "").strip()
            if not tname:
                observation = {"error": "describe_table 需要 arguments.table"}
            else:
                observation = sm.describe_table_graph(tname)
        elif tool == "execute_sql":
            sql = str(args.get("sql") or "").strip()
            if sql_tool_calls >= MAX_SQL_TOOL_CALLS:
                observation = {
                    "error": f"execute_sql 调用次数已达上限 {MAX_SQL_TOOL_CALLS}，请改用 final_answer 总结或缩小问题范围。",
                }
            else:
                sql_tool_calls += 1
                observation = sm.execute_readonly_sql(sql, mode)
        elif tool == "get_foreign_keys":
            t_opt = args.get("table")
            tname = str(t_opt).strip() if t_opt else None
            observation = sm.get_foreign_keys_graph(tname or None)
        else:
            observation = {"error": f"未知工具: {tool!r}，允许: search_tables, describe_table, execute_sql, get_foreign_keys, final_answer"}

        log.info(
            "ReAct step=%s tool=%s args_keys=%s",
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
                "content": "观察结果（JSON，勿重复已成功的无意义调用）：\n"
                + json.dumps(observation, ensure_ascii=False, default=str),
            }
        )

    if final_answer is None:
        final_answer = "（未在步数内给出 final_answer；请增大 max_steps 或简化问题。）"

    return {
        "agent_mode": "react",
        "mode": mode,
        "final_answer": final_answer,
        "trace": trace,
        "sql_tool_calls": sql_tool_calls,
        "steps_used": len(trace),
    }
