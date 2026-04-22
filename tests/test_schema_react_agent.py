"""schema_react_agent 单测（mock LLM，无外部服务）。"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema_react_agent import _parse_react_json, run_schema_react


def test_parse_react_json_fenced():
    raw = '```json\n{"thought": "x", "tool": "final_answer", "arguments": {"answer": "y"}}\n```'
    d = _parse_react_json(raw)
    assert d["tool"] == "final_answer"
    assert d["arguments"]["answer"] == "y"


def test_parse_react_json_invalid_raises():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _parse_react_json("this is not json")


def test_run_schema_react_immediate_final_answer():
    sm = MagicMock()
    out = run_schema_react(
        sm,
        "hello",
        mode="dry_run",
        max_steps=5,
        llm_chat=lambda _: json.dumps(
            {"thought": "无需工具", "tool": "final_answer", "arguments": {"answer": "你好"}}
        ),
    )
    assert out["agent_mode"] == "react"
    assert out["final_answer"] == "你好"
    assert out["sql_tool_calls"] == 0
    sm.search_schema_tables.assert_not_called()


def test_run_schema_react_search_then_final():
    sm = MagicMock()
    sm.search_schema_tables.return_value = {
        "tables": [{"table_name": "users", "content_excerpt": "…", "confidence": 0.6}],
        "empty": False,
    }
    replies = [
        json.dumps(
            {"thought": "找表", "tool": "search_tables", "arguments": {"query": "用户"}}
        ),
        json.dumps({"thought": "够了", "tool": "final_answer", "arguments": {"answer": "有 users 表"}}),
    ]
    idx = {"i": 0}

    def llm(_msgs):
        r = replies[idx["i"]]
        idx["i"] += 1
        return r

    out = run_schema_react(sm, "有哪些用户表", mode="dry_run", max_steps=8, llm_chat=llm)
    assert "users" in out["final_answer"] or out["final_answer"] == "有 users 表"
    sm.search_schema_tables.assert_called_once()
    assert out["sql_tool_calls"] == 0


def test_run_schema_react_execute_sql_respects_dry_run():
    sm = MagicMock()
    sm.execute_readonly_sql.return_value = {"skipped": True, "message": "dry", "sql": "SELECT 1"}
    replies = [
        json.dumps({"thought": "跑 SQL", "tool": "execute_sql", "arguments": {"sql": "SELECT 1"}}),
        json.dumps({"thought": "结束", "tool": "final_answer", "arguments": {"answer": "校验完成"}}),
    ]
    idx = {"i": 0}

    def llm(_msgs):
        r = replies[idx["i"]]
        idx["i"] += 1
        return r

    out = run_schema_react(sm, "q", mode="dry_run", max_steps=6, llm_chat=llm)
    sm.execute_readonly_sql.assert_called_once_with("SELECT 1", "dry_run")
    assert out["sql_tool_calls"] == 1


def test_run_schema_react_unknown_tool_then_final():
    sm = MagicMock()
    replies = [
        json.dumps({"thought": "试", "tool": "not_a_real_tool", "arguments": {}}),
        json.dumps({"thought": "改", "tool": "final_answer", "arguments": {"answer": "已处理未知工具错误"}}),
    ]
    idx = {"i": 0}

    def llm(_msgs):
        r = replies[idx["i"]]
        idx["i"] += 1
        return r

    out = run_schema_react(sm, "q", mode="dry_run", max_steps=6, llm_chat=llm)
    assert "未知工具" in str(out["trace"][0].get("observation", {}))
    assert "已处理" in out["final_answer"]


def test_run_schema_react_describe_table_tool():
    sm = MagicMock()
    sm.describe_table_graph.return_value = {"table": {"name": "users", "columns": []}}
    replies = [
        json.dumps({"thought": "看结构", "tool": "describe_table", "arguments": {"table": "users"}}),
        json.dumps({"thought": "结束", "tool": "final_answer", "arguments": {"answer": "有 users"}}),
    ]
    idx = {"i": 0}

    def llm(_msgs):
        r = replies[idx["i"]]
        idx["i"] += 1
        return r

    out = run_schema_react(sm, "q", mode="dry_run", max_steps=5, llm_chat=llm)
    sm.describe_table_graph.assert_called_once_with("users")
    assert "users" in out["final_answer"]


def test_run_schema_react_max_steps_exhausted():
    sm = MagicMock()
    sm.search_schema_tables.return_value = {"tables": [], "empty": True}

    def llm(_msgs):
        return json.dumps(
            {"thought": "再找", "tool": "search_tables", "arguments": {"query": "noop"}}
        )

    out = run_schema_react(sm, "q", mode="dry_run", max_steps=2, llm_chat=llm)
    assert out["steps_used"] == 2
    assert "未在步数内" in out["final_answer"]
