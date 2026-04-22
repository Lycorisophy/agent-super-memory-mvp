"""FastAPI 请求体验证（仅 Pydantic 模型，不启动服务）。"""
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_schema_query_request_defaults():
    from main import SchemaQueryRequest

    b = SchemaQueryRequest(query="hello")
    assert b.mode == "dry_run"
    assert b.agent_mode == "single_pass"
    assert b.max_steps == 12


def test_schema_query_request_invalid_mode():
    from main import SchemaQueryRequest

    with pytest.raises(ValidationError):
        SchemaQueryRequest(query="x", mode="invalid")


def test_schema_query_request_invalid_agent_mode():
    from main import SchemaQueryRequest

    with pytest.raises(ValidationError):
        SchemaQueryRequest(query="x", agent_mode="deep_thought")


def test_schema_query_request_react_max_steps_bound():
    from main import SchemaQueryRequest

    with pytest.raises(ValidationError):
        SchemaQueryRequest(query="x", max_steps=0)

    b = SchemaQueryRequest(query="x", agent_mode="react", max_steps=40)
    assert b.max_steps == 40
