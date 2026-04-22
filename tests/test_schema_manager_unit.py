"""schema_manager 纯函数单测（无需 Milvus/Neo4j）。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from schema_manager import is_safe_sql, parse_sql_explanation_from_llm


def test_is_safe_sql_select():
    assert is_safe_sql("SELECT id FROM users WHERE 1=1")
    assert is_safe_sql("select * from t")
    assert is_safe_sql("SHOW TABLES")
    assert is_safe_sql("DESCRIBE users")
    assert is_safe_sql("DESC users")
    assert is_safe_sql("EXPLAIN SELECT 1")


def test_is_safe_sql_rejects_injection():
    assert not is_safe_sql("SELECT 1; DROP TABLE users")
    assert not is_safe_sql("DELETE FROM users")
    assert not is_safe_sql("INSERT INTO t VALUES (1)")
    assert not is_safe_sql("UPDATE t SET x=1")
    assert not is_safe_sql("TRUNCATE TABLE t")
    assert not is_safe_sql("SELECT * FROM t WHERE ';' = ';' ; DROP TABLE t")
    assert not is_safe_sql("GRANT ALL ON db.* TO u")
    assert not is_safe_sql("REVOKE SELECT ON t FROM u")
    assert not is_safe_sql("CALL proc()")
    assert not is_safe_sql("SELECT * FROM t UNION DELETE FROM t")


def test_is_safe_sql_empty_and_wrong_prefix():
    assert not is_safe_sql("")
    assert not is_safe_sql("  ")
    assert not is_safe_sql("CREATE TABLE x (id INT)")
    assert is_safe_sql("WITH x AS (SELECT 1) SELECT * FROM x")


def test_parse_sql_explanation_from_llm():
    text = """SQL: SELECT COUNT(*) FROM orders
EXPLANATION: 统计订单总数。"""
    sql, expl = parse_sql_explanation_from_llm(text)
    assert "SELECT COUNT" in sql
    assert "统计订单" in expl


def test_parse_sql_explanation_from_llm_fenced_sql():
    text = """Here is SQL:
```sql
SELECT 2 AS two
```
EXPLANATION: 取常量。"""
    sql, expl = parse_sql_explanation_from_llm(text)
    assert "SELECT 2" in sql
    assert "常量" in expl or expl


def test_is_safe_sql_trailing_semicolon_ok():
    assert is_safe_sql("SELECT 1 AS x;")


def test_query_schema_dry_run_skips_mysql():
    from schema_manager import SchemaManager

    mem = MagicMock()
    mem._get_embedding.return_value = [0.0] * 4096

    real = object.__new__(SchemaManager)
    real.memory = mem
    real.schema_col = MagicMock()
    real.schema_col.search.return_value = [[]]

    out = SchemaManager.query_schema(real, "hello", mode="dry_run")
    assert "error" in out

    with patch("schema_manager.create_engine") as ce:
        real2 = object.__new__(SchemaManager)
        real2.memory = mem
        real2.schema_col = MagicMock()
        hit = MagicMock()
        hit.entity.get.side_effect = lambda k: "users" if k == "table_name" else "x"
        real2.schema_col.search.return_value = [[hit]]
        mem._driver.session.return_value.__enter__.return_value.run.return_value.single.return_value = {
            "name": "users",
            "comment": "",
            "columns": [{"name": "id", "type": "int", "comment": "", "key": "PRI"}],
        }

        def fake_prompt(q, tables):
            return "p"

        def fake_llm(p):
            return "SQL: SELECT 1 AS one\nEXPLANATION: 测试。"

        real2._build_sql_prompt = fake_prompt  # type: ignore[method-assign]
        real2._call_llm_for_sql_raw = fake_llm  # type: ignore[method-assign]

        out2 = SchemaManager.query_schema(real2, "数一下", mode="dry_run")
        assert out2.get("sql")
        assert "explanation" in out2
        ce.assert_not_called()


def test_execute_readonly_sql_dry_run_no_engine():
    from schema_manager import SchemaManager

    sm = object.__new__(SchemaManager)
    with patch("schema_manager.create_engine") as ce:
        out = SchemaManager.execute_readonly_sql(sm, "SELECT 1", "dry_run")
        assert out.get("skipped") is True
        ce.assert_not_called()


def test_execute_readonly_sql_invalid_mode():
    from schema_manager import SchemaManager

    sm = object.__new__(SchemaManager)
    out = SchemaManager.execute_readonly_sql(sm, "SELECT 1", "wrong")
    assert "error" in out
