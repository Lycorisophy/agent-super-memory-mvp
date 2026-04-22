"""memory_agent 永驻记忆工具单测。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory_agent
from memory_service import MemorySystem


def test_run_tool_update_permanent_success():
    mem = MagicMock(spec=MemorySystem)
    pm = MagicMock()
    pm.upsert.return_value = {"success": True}
    out = memory_agent._run_tool(
        mem,
        "update_permanent_memory",
        {"category": "用户身份", "value": "工程师", "user_id": "ignored"},
        "[t]",
        perm_store=pm,
    )
    assert out == {"success": True}
    pm.upsert.assert_called_once()
    assert pm.upsert.call_args[0][0] == memory_agent.settings.default_user_id


def test_run_tool_update_permanent_no_store():
    mem = MagicMock(spec=MemorySystem)
    out = memory_agent._run_tool(
        mem,
        "update_permanent_memory",
        {"category": "用户身份", "value": "x"},
        "[t]",
        perm_store=None,
    )
    assert out["success"] is False
