"""unified_dialogue_agent 永驻工具单测。"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unified_dialogue_agent import run_unified_dialogue


def test_unified_update_permanent_then_final():
    mem = MagicMock()
    dlg = MagicMock()
    pm = MagicMock()
    pm.format_prompt_block.return_value = "## 永驻…"
    pm.upsert.return_value = {"success": True}
    replies = [
        json.dumps(
            {
                "thought": "改规范",
                "tool": "update_permanent_memory",
                "arguments": {"category": "工作规范", "value": "先测试后上线"},
            }
        ),
        json.dumps({"thought": "答", "tool": "final_answer", "arguments": {"answer": "已更新"}}),
    ]
    idx = {"i": 0}

    def llm(_msgs):
        r = replies[idx["i"]]
        idx["i"] += 1
        return r

    out = run_unified_dialogue(
        mem, dlg, "hi", [], user_id="u9", max_steps=8, llm_chat=llm, perm_store=pm
    )
    assert "已更新" in out["final_answer"]
    pm.upsert.assert_called_once_with("u9", "工作规范", "先测试后上线")
