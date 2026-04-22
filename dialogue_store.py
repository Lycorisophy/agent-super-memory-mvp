"""
MySQL 对话表：最近 N 条、按 id 游标查更早、写入 user/assistant 两行。
表名与列名来自配置并校验为合法标识符；值一律参数化绑定。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config import settings

log = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_sql_identifier(name: str, *, what: str) -> str:
    s = (name or "").strip()
    if not _IDENT_RE.match(s):
        raise ValueError(f"非法 {what}: {name!r}（仅允许字母、数字、下划线，且不以数字开头）")
    return s


def _quote_ident(name: str, *, what: str) -> str:
    s = _validate_sql_identifier(name, what=what)
    return f"`{s}`"


class DialogueStore:
    """对话表访问；表名白名单即配置中的单表名，经标识符校验。"""

    def __init__(
        self,
        engine: Engine,
        *,
        table: str,
        col_id: str,
        col_user_id: str,
        col_role: str,
        col_content: str,
        col_created_at: str,
    ) -> None:
        self.engine = engine
        self._t = _quote_ident(table, what="dialogue_table")
        self._id = _quote_ident(col_id, what="dialogue_col_id")
        self._uid = _quote_ident(col_user_id, what="dialogue_col_user_id")
        self._role = _quote_ident(col_role, what="dialogue_col_role")
        self._content = _quote_ident(col_content, what="dialogue_col_content")
        self._created = _quote_ident(col_created_at, what="dialogue_col_created_at")

    @classmethod
    def from_settings(cls) -> "DialogueStore":
        raw_url = (settings.mysql_dialogue_url or "").strip()
        url = raw_url or (settings.mysql_url or "").strip()
        if not url:
            raise ValueError("未配置 mysql_dialogue_url 且 mysql_url 为空，无法连接对话库")

        engine = create_engine(url, pool_pre_ping=True)
        return cls(
            engine,
            table=settings.dialogue_table,
            col_id=settings.dialogue_col_id,
            col_user_id=settings.dialogue_col_user_id,
            col_role=settings.dialogue_col_role,
            col_content=settings.dialogue_col_content,
            col_created_at=settings.dialogue_col_created_at,
        )

    def ping(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def fetch_recent(self, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """按 id 降序取 limit 条，再反转为时间正序（旧 → 新）。"""
        lim = max(1, min(int(limit), 500))
        uid = (user_id or "").strip()
        sql = (
            f"SELECT {self._id} AS id, {self._uid} AS user_id, {self._role} AS role, "
            f"{self._content} AS content, {self._created} AS created_at "
            f"FROM {self._t} WHERE {self._uid} = :uid ORDER BY {self._id} DESC LIMIT :lim"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), {"uid": uid, "lim": lim}).mappings().all()
        out = [dict(r) for r in reversed(rows)]
        return out

    def fetch_older(
        self,
        user_id: str,
        before_id: int,
        limit: int,
        *,
        older_max: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """id 严格小于 before_id，按 id 降序取至多 limit 条，返回为正序（旧 → 新）。"""
        cap = older_max if older_max is not None else settings.dialogue_older_fetch_max
        lim = max(1, min(int(limit), int(cap)))
        bid = int(before_id)
        uid = (user_id or "").strip()
        sql = (
            f"SELECT {self._id} AS id, {self._uid} AS user_id, {self._role} AS role, "
            f"{self._content} AS content, {self._created} AS created_at "
            f"FROM {self._t} WHERE {self._uid} = :uid AND {self._id} < :bid "
            f"ORDER BY {self._id} DESC LIMIT :lim"
        )
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), {"uid": uid, "bid": bid, "lim": lim}).mappings().all()
        return [dict(r) for r in reversed(rows)]

    def append_exchange(self, user_id: str, user_content: str, assistant_content: str) -> None:
        """先插入 user，再插入 assistant（同一连接内顺序执行）。"""
        uid = (user_id or "").strip()
        sql = (
            f"INSERT INTO {self._t} ({self._uid}, {self._role}, {self._content}) "
            f"VALUES (:uid, :role, :content)"
        )
        stmt = text(sql)
        with self.engine.begin() as conn:
            conn.execute(stmt, {"uid": uid, "role": "user", "content": user_content})
            conn.execute(stmt, {"uid": uid, "role": "assistant", "content": assistant_content})
        log.info("dialogue_store: appended user+assistant rows user_id=%s", uid)
