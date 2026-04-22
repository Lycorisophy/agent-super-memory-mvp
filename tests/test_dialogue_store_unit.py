"""dialogue_store 单测（无真实 MySQL）。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialogue_store import DialogueStore, _validate_sql_identifier


def test_validate_sql_identifier_ok():
    assert _validate_sql_identifier("chat_messages", what="table") == "chat_messages"


def test_validate_sql_identifier_rejects_injection():
    with pytest.raises(ValueError):
        _validate_sql_identifier("x;DROP TABLE t", what="table")


def test_fetch_recent_reverses_to_chronological_order():
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {"id": 3, "user_id": "u1", "role": "assistant", "content": "c", "created_at": None},
        {"id": 1, "user_id": "u1", "role": "user", "content": "a", "created_at": None},
    ]
    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_result
    cm = MagicMock()
    cm.__enter__.return_value = mock_conn
    cm.__exit__.return_value = None
    mock_engine = MagicMock()
    mock_engine.connect.return_value = cm

    store = DialogueStore(
        mock_engine,
        table="chat_messages",
        col_id="id",
        col_user_id="user_id",
        col_role="role",
        col_content="content",
        col_created_at="created_at",
    )
    out = store.fetch_recent("u1", 10)
    assert [r["id"] for r in out] == [1, 3]
    sql_arg = mock_conn.execute.call_args[0][0]
    assert "FROM `chat_messages`" in str(sql_arg)
    assert ":uid" in str(sql_arg)


def test_append_exchange_two_inserts():
    mock_conn = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = mock_conn
    cm.__exit__.return_value = None
    mock_engine = MagicMock()
    mock_engine.begin.return_value = cm

    store = DialogueStore(
        mock_engine,
        table="chat_messages",
        col_id="id",
        col_user_id="user_id",
        col_role="role",
        col_content="content",
        col_created_at="created_at",
    )
    store.append_exchange("u1", "hello", "world")
    assert mock_conn.execute.call_count == 2
