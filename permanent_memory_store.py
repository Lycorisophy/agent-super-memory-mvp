"""
MySQL 永驻记忆：每用户三条（用户身份、智能体性格、工作规范），每项最多 1000 字。
连接串与对话库一致：mysql_dialogue_url 非空则用之，否则 mysql_url（须 SELECT + INSERT/UPDATE）。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import settings

log = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

MAX_PERMANENT_CONTENT_LEN = 1000

# 工具/API 入参（中文） -> 库内 category 值
PERMANENT_LABEL_TO_KEY: Dict[str, str] = {
    "用户身份": "user_identity",
    "智能体性格": "agent_personality",
    "工作规范": "work_norms",
}

PERMANENT_DISPLAY_ORDER: List[Tuple[str, str]] = [
    ("用户身份", "user_identity"),
    ("智能体性格", "agent_personality"),
    ("工作规范", "work_norms"),
]


def _validate_sql_identifier(name: str, *, what: str) -> str:
    s = (name or "").strip()
    if not _IDENT_RE.match(s):
        raise ValueError(f"非法 {what}: {name!r}（仅允许字母、数字、下划线，且不以数字开头）")
    return s


def _quote_ident(name: str, *, what: str) -> str:
    return f"`{_validate_sql_identifier(name, what=what)}`"


def normalize_permanent_category_label(label: str) -> str:
    """中文类别 -> 内部 category 键；非法则抛 ValueError。"""
    s = (label or "").strip()
    if s not in PERMANENT_LABEL_TO_KEY:
        raise ValueError(
            f"类别须为以下之一：{', '.join(PERMANENT_LABEL_TO_KEY.keys())}，收到: {label!r}"
        )
    return PERMANENT_LABEL_TO_KEY[s]


class PermanentMemoryStore:
    def __init__(self, engine: Engine, *, table: str) -> None:
        self.engine = engine
        self._t = _quote_ident(table, what="permanent_memory_table")
        self._uid = _quote_ident("user_id", what="column")
        self._cat = _quote_ident("category", what="column")
        self._content = _quote_ident("content", what="column")
        self._updated = _quote_ident("updated_at", what="column")

    @classmethod
    def from_settings(cls) -> "PermanentMemoryStore":
        raw_url = (settings.mysql_dialogue_url or "").strip()
        url = raw_url or (settings.mysql_url or "").strip()
        if not url:
            raise ValueError("未配置 mysql_dialogue_url 且 mysql_url 为空，无法连接永驻记忆库")
        engine = create_engine(url, pool_pre_ping=True)
        return cls(engine, table=settings.permanent_memory_table)

    def ping(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def load_all(self, user_id: str) -> Dict[str, str]:
        """返回 category_key -> content；无行则该键不存在。"""
        uid = (user_id or "").strip()
        sql = (
            f"SELECT {self._cat} AS category, {self._content} AS content "
            f"FROM {self._t} WHERE {self._uid} = :uid"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), {"uid": uid}).mappings().all()
        out: Dict[str, str] = {}
        for r in rows:
            k = (r.get("category") or "").strip()
            if k in {x[1] for x in PERMANENT_DISPLAY_ORDER}:
                out[k] = (r.get("content") or "")[:MAX_PERMANENT_CONTENT_LEN]
        return out

    def format_prompt_block(self, user_id: str) -> str:
        data = self.load_all(user_id)
        lines = ["## 永驻记忆（可经工具 update_permanent_memory 修改；仅从下列三类读取）"]
        for label, key in PERMANENT_DISPLAY_ORDER:
            body = (data.get(key) or "").strip()
            lines.append(f"- **{label}**：{body if body else '（未设置）'}")
        return "\n".join(lines)

    def upsert(self, user_id: str, category_label_cn: str, value: str) -> Dict[str, Any]:
        """
        按中文类别写入/更新。value 超过 1000 字返回 success: false。
        """
        try:
            cat_key = normalize_permanent_category_label(category_label_cn)
        except ValueError as e:
            return {"success": False, "error": str(e)}
        raw = value if isinstance(value, str) else str(value)
        content = raw.strip()
        if len(content) > MAX_PERMANENT_CONTENT_LEN:
            return {
                "success": False,
                "error": f"内容超过 {MAX_PERMANENT_CONTENT_LEN} 字",
            }
        uid = (user_id or "").strip()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        sql = (
            f"INSERT INTO {self._t} ({self._uid}, {self._cat}, {self._content}, {self._updated}) "
            f"VALUES (:uid, :cat, :content, :ts) "
            f"ON DUPLICATE KEY UPDATE {self._content} = VALUES({self._content}), "
            f"{self._updated} = VALUES({self._updated})"
        )
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    text(sql),
                    {"uid": uid, "cat": cat_key, "content": content, "ts": now},
                )
        except Exception as e:
            log.warning("永驻记忆 upsert 失败: %s", e)
            return {"success": False, "error": str(e)}
        log.info("永驻记忆已更新 user_id=%s category=%s len=%d", uid, cat_key, len(content))
        return {"success": True}
