"""v5 tense/confidence 枚举校验（无外部服务）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_service import (
    CONFIDENCE_ALLOWED,
    TENSE_ALLOWED,
    _normalize_optional_enum,
)


def test_normalize_tense():
    assert _normalize_optional_enum("past", TENSE_ALLOWED) == "past"
    assert _normalize_optional_enum("PRESENT", TENSE_ALLOWED) == "present"
    assert _normalize_optional_enum("invalid", TENSE_ALLOWED) is None
    assert _normalize_optional_enum(None, TENSE_ALLOWED) is None
    assert _normalize_optional_enum("", TENSE_ALLOWED) is None


def test_normalize_confidence():
    assert _normalize_optional_enum("real", CONFIDENCE_ALLOWED) == "real"
    assert _normalize_optional_enum("Planned", CONFIDENCE_ALLOWED) == "planned"
    assert _normalize_optional_enum("wrong", CONFIDENCE_ALLOWED) is None


if __name__ == "__main__":
    test_normalize_tense()
    test_normalize_confidence()
    print("v5 enum checks ok")
