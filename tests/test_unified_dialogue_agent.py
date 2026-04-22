"""unified_dialogue_agent 单测（mock LLM / MemorySystem / DialogueStore）。"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_dialogue_agent import run_unified_dialogue


def test_run_unified_immediate_final_answer():
    mem = MagicMock()
    dlg = MagicMock()
    dlg.fetch_recent.return_value = []
    out = run_unified_dialogue(
        mem,
        dlg,
        "你好",
        [],
        user_id="tester",
        max_steps=5,
        llm_chat=lambda _msgs: json.dumps(
            {"thought": "直接答", "tool": "final_answer", "arguments": {"answer": "你好呀"}}
        ),
    )
    assert out["agent_mode"] == "unified_react"
    assert out["final_answer"] == "你好呀"
    assert out["fetch_older_chat_calls"] == 0
    mem.store_memory.assert_not_called()


def test_store_memory_delegates_to_memory_system():
    mem = MagicMock()
    mem.store_memory.return_value = {"ok": True, "memory_ids": ["m1"]}
    dlg = MagicMock()
    replies = [
        json.dumps(
            {
                "thought": "写入",
                "tool": "store_memory",
                "arguments": {
                    "memories": [
                        {
                            "type": "fact",
                            "content": "键:值测试",
                            "time": "2020-01-01 12:00:00",
                        }
                    ],
                    "relations": [],
                },
            }
        ),
        json.dumps({"thought": "结束", "tool": "final_answer", "arguments": {"answer": "已记"}}),
    ]
    idx = {"i": 0}

    def llm(_msgs):
        r = replies[idx["i"]]
        idx["i"] += 1
        return r

    out = run_unified_dialogue(
        mem, dlg, "我的爱好是下棋", [], user_id="u1", max_steps=8, llm_chat=llm
    )
    assert out["final_answer"] == "已记"
    mem.store_memory.assert_called_once()
    call_kw = mem.store_memory.call_args
    assert call_kw[0][0] == "u1"
    assert call_kw[1]["memories"][0]["type"] == "fact"


def test_fetch_older_chat_calls_dialogue_store():
    mem = MagicMock()
    dlg = MagicMock()
    dlg.fetch_older.return_value = [{"id": 5, "role": "user", "content": "旧消息"}]
    replies = [
        json.dumps(
            {
                "thought": "看更早",
                "tool": "fetch_older_chat",
                "arguments": {"before_id": 100, "limit": 3},
            }
        ),
        json.dumps({"thought": "答", "tool": "final_answer", "arguments": {"answer": "读到了"}}),
    ]
    idx = {"i": 0}

    def llm(_msgs):
        r = replies[idx["i"]]
        idx["i"] += 1
        return r

    out = run_unified_dialogue(mem, dlg, "继续", [], user_id="u2", max_steps=8, llm_chat=llm)
    assert "读到了" in out["final_answer"]
    dlg.fetch_older.assert_called_once()
    assert dlg.fetch_older.call_args[0][0] == "u2"
    assert dlg.fetch_older.call_args[0][1] == 100


def test_query_memory_delegates():
    mem = MagicMock()
    mem.query_memory.return_value = {"memories": [], "causal_chain": []}
    dlg = MagicMock()
    replies = [
        json.dumps(
            {
                "thought": "查记忆",
                "tool": "query_memory",
                "arguments": {"query_text": "爱好", "top_k": 3},
            }
        ),
        json.dumps({"thought": "答", "tool": "final_answer", "arguments": {"answer": "无"}}),
    ]
    idx = {"i": 0}

    def llm(_msgs):
        r = replies[idx["i"]]
        idx["i"] += 1
        return r

    out = run_unified_dialogue(mem, dlg, "问", [], user_id="u3", max_steps=8, llm_chat=llm)
    assert "无" in out["final_answer"]
    mem.query_memory.assert_called_once()
    assert mem.query_memory.call_args[0][1] == "爱好"
