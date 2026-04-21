"""design v4 纯函数与解析逻辑（无需 Milvus/Neo4j）。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_service import (
    MemoryType,
    _build_unified_memory_content,
    _parse_fact_kv_from_core,
    _parse_time_string_to_ts,
)


def test_build_unified_content():
    s = _build_unified_memory_content(
        MemoryType.fact.value,
        "2026-04-21 15:00",
        "",
        "我",
        "居住城市 = 杭州",
        "用户说搬到杭州了",
    )
    assert "【类型】事实" in s
    assert "居住城市 = 杭州" in s
    assert "【来源】用户说搬到杭州了" in s


def test_parse_fact_kv():
    assert _parse_fact_kv_from_core("配偶姓名 = 李丽") == ("配偶姓名", "李丽")
    assert _parse_fact_kv_from_core("no_equals") is None


def test_parse_time_roundtrip():
    disp, ts = _parse_time_string_to_ts("2026-01-15 08:30")
    assert "2026-01-15" in disp
    assert ts > int(time.time()) - 86400 * 400


if __name__ == "__main__":
    test_build_unified_content()
    test_parse_fact_kv()
    test_parse_time_roundtrip()
    print("v4 unit checks ok")
