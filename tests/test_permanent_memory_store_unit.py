"""permanent_memory_store 单测（无真实 MySQL）。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from permanent_memory_store import (
    MAX_PERMANENT_CONTENT_LEN,
    PermanentMemoryStore,
    normalize_permanent_category_label,
)


def test_normalize_category_ok():
    assert normalize_permanent_category_label("  用户身份  ") == "user_identity"


def test_normalize_category_invalid():
    with pytest.raises(ValueError):
        normalize_permanent_category_label("其它")


def test_upsert_rejects_too_long():
    eng = MagicMock()
    store = PermanentMemoryStore(
        eng,
        table="permanent_memory",
    )
    out = store.upsert("u1", "工作规范", "x" * (MAX_PERMANENT_CONTENT_LEN + 1))
    assert out["success"] is False
    assert "1000" in out.get("error", "")


def test_format_prompt_block_empty():
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = []
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_result
    cm = MagicMock()
    cm.__enter__.return_value = mock_conn
    cm.__exit__.return_value = None
    eng = MagicMock()
    eng.connect.return_value = cm

    store = PermanentMemoryStore(eng, table="permanent_memory")
    text = store.format_prompt_block("u1")
    assert "用户身份" in text
    assert "（未设置）" in text


def test_upsert_executes_insert():
    mock_conn = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = mock_conn
    cm.__exit__.return_value = None
    eng = MagicMock()
    eng.begin.return_value = cm

    store = PermanentMemoryStore(eng, table="permanent_memory")
    out = store.upsert("u1", "智能体性格", "温和")
    assert out["success"] is True
    assert mock_conn.execute.call_count == 1
    arg0 = str(mock_conn.execute.call_args[0][0])
    assert "ON DUPLICATE KEY UPDATE" in arg0
    assert "`permanent_memory`" in arg0
