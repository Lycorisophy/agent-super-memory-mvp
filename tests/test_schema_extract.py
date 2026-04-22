"""表信息提取：规范化、Markdown 关联列、Excel、自然语言 JSON finalize（无 Milvus/Neo4j）。"""
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema_manager import (
    SchemaManager,
    _dedupe_relations,
    _finalize_table_dict,
    _finalize_tables_list,
    _is_md_separator_row,
    _markdown_header_indices,
    _relation_from_link_cell,
    _strip_json_fence,
    normalize_schema_identifier,
)


def test_normalize_schema_identifier():
    assert normalize_schema_identifier("User-Name") == "user_name"
    assert normalize_schema_identifier("  Orders  ") == "orders"


def test_relation_from_link_cell():
    r = _relation_from_link_cell("user_id", "users.id")
    assert r is not None
    assert r["target_table"] == "users"
    assert r["source_cols"] == ["user_id"]
    assert r["target_cols"] == ["id"]
    assert _relation_from_link_cell("x", "bad") is None


def test_finalize_table_dict():
    raw = {
        "name": "Orders",
        "comment": "订单",
        "columns": [{"name": "ID", "type": "INT", "nullable": False, "key": "pri", "comment": "主键"}],
        "relations": [
            {
                "type": "belongs_to",
                "target_table": "Users",
                "source_cols": ["user_id"],
                "target_cols": ["id"],
            }
        ],
    }
    t = _finalize_table_dict(raw)
    assert t["name"] == "orders"
    assert t["columns"][0]["name"] == "id"
    assert t["columns"][0]["key"] == "PRI"
    assert t["relations"][0]["target_table"] == "users"


def test_parse_markdown_with_header_and_link():
    md = """## orders（订单）

| 字段名 | 类型 | 注释 | 关联 |
| ------ | ---- | ---- | ---- |
| id | BIGINT | 主键 | |
| user_id | BIGINT | 用户 | users.id |
| amount | DECIMAL | 金额 | |
"""
    sm = object.__new__(SchemaManager)
    out = SchemaManager.parse_markdown(sm, md)
    assert len(out) == 1
    assert out[0]["name"] == "orders"
    assert any(c["name"] == "user_id" for c in out[0]["columns"])
    rels = out[0].get("relations") or []
    assert len(rels) >= 1
    assert rels[0]["target_table"] == "users"


def test_parse_markdown_legacy_three_column():
    md = Path(__file__).resolve().parent.joinpath("schema_sample_users.md").read_text(encoding="utf-8")
    sm = object.__new__(SchemaManager)
    out = SchemaManager.parse_markdown(sm, md)
    assert len(out) >= 1
    assert out[0]["name"] == "users"


def test_finalize_tables_list_filters_empty():
    assert _finalize_tables_list([{"name": "x", "columns": []}]) == []


def test_dedupe_relations():
    r = {"type": "belongs_to", "target_table": "users", "source_cols": ["a"], "target_cols": ["id"]}
    assert len(_dedupe_relations([r, dict(r)])) == 1


def test_is_md_separator_row():
    assert _is_md_separator_row(["---", "---", "---"])
    assert _is_md_separator_row([":--", ":-:"])
    assert not _is_md_separator_row(["字段名", "类型"])


def test_markdown_header_indices_english():
    cells = ["Field", "Type", "Comment", "Key", "Relation"]
    idx = _markdown_header_indices(cells)
    assert idx.get("name") == 0 and idx.get("type") == 1 and idx.get("link") == 4


def test_strip_json_fence():
    raw = "```json\n[1]\n```"
    assert _strip_json_fence(raw) == "[1]"


def test_parse_markdown_multiple_tables():
    md = """## accounts

| 字段名 | 类型 | 注释 |
| --- | --- | --- |
| id | INT | x |

## brands

| Field | Type | Comment |
| --- | --- | --- |
| id | INT | y |
"""
    sm = object.__new__(SchemaManager)
    out = SchemaManager.parse_markdown(sm, md)
    names = {t["name"] for t in out}
    assert names == {"accounts", "brands"}


def test_parse_excel_with_relation_column():
    df = pd.DataFrame(
        [
            {"字段名": "id", "类型": "INT", "注释": "主键", "键": "PRI", "关联关系": ""},
            {
                "字段名": "user_id",
                "类型": "BIGINT",
                "注释": "用户",
                "键": "MUL",
                "关联关系": "users.id",
            },
        ]
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="orders", index=False)
    raw = buf.getvalue()
    sm = object.__new__(SchemaManager)
    out = SchemaManager.parse_excel(sm, raw)
    assert len(out) == 1
    assert out[0]["name"] == "orders"
    rels = out[0].get("relations") or []
    assert any(r["target_table"] == "users" for r in rels)


def test_parse_excel_english_columns():
    df = pd.DataFrame(
        [
            {"Field": "id", "Type": "INT", "Comment": "pk", "Key": "PRI"},
        ]
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="items", index=False)
    sm = object.__new__(SchemaManager)
    out = SchemaManager.parse_excel(sm, buf.getvalue())
    assert len(out) == 1
    assert out[0]["name"] == "items"
    assert out[0]["columns"][0]["name"] == "id"


def test_parse_natural_language_tables_json_success():
    sm = object.__new__(SchemaManager)
    sm._call_llm_for_sql_raw = MagicMock(  # type: ignore[method-assign]
        return_value=json.dumps(
            [
                {
                    "name": "widgets",
                    "comment": "",
                    "columns": [{"name": "id", "type": "BIGINT", "nullable": False, "key": "PRI", "comment": ""}],
                    "relations": [],
                }
            ],
            ensure_ascii=False,
        )
    )
    out = SchemaManager.parse_natural_language_tables(sm, "请描述 widgets 表只有 id 主键")
    assert len(out) == 1
    assert out[0]["name"] == "widgets"


def test_parse_natural_language_tables_retry_on_bad_json():
    sm = object.__new__(SchemaManager)
    good = json.dumps(
        [{"name": "t", "comment": "", "columns": [{"name": "id", "type": "INT"}], "relations": []}]
    )
    sm._call_llm_for_sql_raw = MagicMock(side_effect=["not json", good])  # type: ignore[method-assign]
    out = SchemaManager.parse_natural_language_tables(sm, "描述表 t")
    assert len(out) == 1
    assert sm._call_llm_for_sql_raw.call_count == 2


def test_parse_natural_language_tables_empty_input():
    sm = object.__new__(SchemaManager)
    sm._call_llm_for_sql_raw = MagicMock()  # type: ignore[method-assign]
    assert SchemaManager.parse_natural_language_tables(sm, "   ") == []
    sm._call_llm_for_sql_raw.assert_not_called()


def test_finalize_table_dict_invalid_key_becomes_empty():
    t = _finalize_table_dict(
        {
            "name": "t",
            "columns": [{"name": "c", "type": "INT", "key": "BOGUS", "comment": ""}],
        }
    )
    assert t["columns"][0]["key"] == ""


def test_excel_resolve_columns_requires_comment():
    sm = object.__new__(SchemaManager)
    df = pd.DataFrame([{"Field": "id", "Type": "INT"}])
    assert SchemaManager._excel_resolve_columns(sm, df) is None


def test_finalize_table_dict_relation_requires_cols():
    t = _finalize_table_dict(
        {
            "name": "t",
            "columns": [{"name": "id", "type": "INT"}],
            "relations": [{"type": "belongs_to", "target_table": "u", "source_cols": [], "target_cols": ["id"]}],
        }
    )
    assert t["relations"] == []
