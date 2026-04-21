"""
长期记忆存储与查询（Milvus + Neo4j），实现见 design.md v4.0 + v5.0 增量（:Memory 上 tense/confidence）。
旧版 memory_chunks / Event·Fact·Knowledge 与 v4 不兼容，需使用新集合名与新 Neo4j 约束。
"""
from __future__ import annotations

import logging
import random
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

import ollama
from neo4j import GraphDatabase
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from config import settings


class MemoryType(str, Enum):
    event = "event"
    fact = "fact"
    knowledge = "knowledge"


_MEMORY_TYPE_CN = {
    MemoryType.event.value: "事件",
    MemoryType.fact.value: "事实",
    MemoryType.knowledge.value: "知识",
}

TENSE_ALLOWED = frozenset({"past", "present", "future"})
CONFIDENCE_ALLOWED = frozenset({"real", "imagined", "planned"})


def _normalize_optional_enum(value: Any, allowed: frozenset) -> Optional[str]:
    """v5：可选枚举；非法或空返回 None。"""
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s not in allowed:
        log.debug("忽略非法枚举值 %r（允许 %s）", value, sorted(allowed))
        return None
    return s


class SnowflakeIDGenerator:
    """64 位雪花 ID（毫秒时间戳 + 机器号 + 序列）。"""

    def __init__(self, datacenter_id: int = 1, worker_id: int = 1):
        self.datacenter_id = datacenter_id & 0x1F
        self.worker_id = worker_id & 0x1F
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()
        self.twepoch = 1704067200000

    def _current_millis(self) -> int:
        return int(time.time() * 1000)

    def _wait_next_millis(self, last_timestamp: int) -> int:
        timestamp = self._current_millis()
        while timestamp <= last_timestamp:
            timestamp = self._current_millis()
        return timestamp

    def next_id(self) -> int:
        with self.lock:
            timestamp = self._current_millis()
            if timestamp < self.last_timestamp:
                raise RuntimeError("Clock moved backwards")
            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & 0xFFF
                if self.sequence == 0:
                    timestamp = self._wait_next_millis(self.last_timestamp)
            else:
                self.sequence = 0
            self.last_timestamp = timestamp
            return (
                ((timestamp - self.twepoch) << 22)
                | (self.datacenter_id << 17)
                | (self.worker_id << 12)
                | self.sequence
            )

    def next_str(self) -> str:
        return str(self.next_id())


def _milvus_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _milvus_collection_embedding_dim(col: Collection) -> Optional[int]:
    """已存在集合中 FLOAT_VECTOR 字段的维度；新建集合前无 schema 时不调用。"""
    try:
        for field in col.schema.fields:
            if field.dtype == DataType.FLOAT_VECTOR:
                params = getattr(field, "params", None) or {}
                d = params.get("dim")
                if d is not None:
                    return int(d)
                d2 = getattr(field, "dim", None)
                if d2 is not None:
                    return int(d2)
    except Exception as e:
        log.warning("读取 Milvus 向量维度失败: %s", e)
    return None


def _assert_milvus_dim_matches_settings(col: Collection) -> None:
    """已有集合的 embedding 维须与 settings.vector_dim 一致，否则插入阶段才会报难懂的除法错误。"""
    actual = _milvus_collection_embedding_dim(col)
    if actual is None:
        return
    expected = int(settings.vector_dim)
    if actual != expected:
        name = settings.milvus_collection
        raise RuntimeError(
            f"Milvus 集合「{name}」中向量字段维度为 {actual}，与当前配置 vector_dim={expected} 不一致"
            f"（常见于曾用 768 维建库、后改回 Qwen 4096 维嵌入）。"
            f"请任选其一：1) 在 Milvus 中删除集合 {name} 后重启服务以按 vector_dim={expected} 重建；"
            f"2) 或在 .env 将 VECTOR_DIM 改为 {actual} 并换用输出 {actual} 维的嵌入模型。"
        )


def _format_time_display(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _parse_time_string_to_ts(time_str: Optional[str]) -> Tuple[str, int]:
    """返回 (YYYY-MM-DD HH:MM 展示串, Unix 秒)。"""
    s = (time_str or "").strip()
    if not s:
        ts = int(time.time())
        return _format_time_display(ts), ts
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            ts = int(dt.timestamp())
            return dt.strftime("%Y-%m-%d %H:%M"), ts
        except ValueError:
            continue
    ts = int(time.time())
    return _format_time_display(ts), ts


def _build_unified_memory_content(
    memory_type: str,
    time_display: str,
    location: str,
    subject: str,
    core_content: str,
    source: str,
) -> str:
    """design.md §2.1 统一中文模板。"""
    type_cn = _MEMORY_TYPE_CN.get(memory_type, memory_type)
    lines = [
        f"【类型】{type_cn}",
        f"【时间】{time_display}",
        f"【地点】{(location or '').strip()}",
        f"【主体】{(subject or '').strip()}",
        f"【内容】{(core_content or '').strip()}",
        f"【来源】{(source or '').strip()}",
    ]
    return "\n".join(lines)


def _parse_fact_kv_from_core(core: str) -> Optional[Tuple[str, str]]:
    """从核心内容解析「键 = 值」。"""
    line = (core or "").strip()
    if not line or "=" not in line:
        return None
    k, _, v = line.partition("=")
    k, v = k.strip(), v.strip()
    if not k or not v:
        return None
    return k, v


def _memory_payload_from_record(
    rec: Dict[str, Any], memory_id: str
) -> Dict[str, Any]:
    """组装返回给工具的 memories 项（含 v5 可选 tense/confidence）。"""
    item: Dict[str, Any] = {
        "memory_id": memory_id,
        "content": rec["content"],
        "timestamp": rec["timestamp"],
        "memory_type": rec["memory_type"],
    }
    if rec.get("tense") is not None:
        item["tense"] = rec["tense"]
    if rec.get("confidence") is not None:
        item["confidence"] = rec["confidence"]
    return item


def _content_line_summary(content: str, max_len: int = 120) -> str:
    """从统一文本中取【内容】一行作因果链摘要。"""
    for raw in (content or "").splitlines():
        line = raw.strip()
        if line.startswith("【内容】"):
            inner = line.replace("【内容】", "", 1).strip()
            return inner[:max_len] if inner else line[:max_len]
    return (content or "").replace("\n", " ")[:max_len]


class MemorySystem:
    """连接 Milvus / Neo4j，提供 v4/v5 store_memory 与 query_memory。"""

    def __init__(self) -> None:
        self._id_gen = SnowflakeIDGenerator(
            settings.snowflake_datacenter_id, settings.snowflake_worker_id
        )
        self._collection: Optional[Collection] = None
        self._driver = None
        self._ollama_host = settings.ollama_base_url

    def connect(self) -> None:
        log.info(
            "MemorySystem.connect: Milvus %s:%s collection=%s dim=%s",
            settings.milvus_host,
            settings.milvus_port,
            settings.milvus_collection,
            settings.vector_dim,
        )
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        self._collection = self._init_milvus_collection()
        _assert_milvus_dim_matches_settings(self._collection)
        log.info("MemorySystem.connect: Milvus 集合已加载")
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        log.info("MemorySystem.connect: Neo4j 驱动已创建 uri=%s", settings.neo4j_uri)
        self._ensure_neo4j_constraints()
        log.info("MemorySystem.connect: Neo4j 约束已检查")

    def close(self) -> None:
        log.info("MemorySystem.close: 正在断开")
        if self._driver:
            self._driver.close()
            self._driver = None
        try:
            connections.disconnect("default")
        except Exception:
            pass
        self._collection = None
        log.info("MemorySystem.close: 完成")

    def _get_embedding(self, text: str) -> List[float]:
        last_err: Optional[BaseException] = None
        for attempt in range(max(1, settings.ollama_embed_retries)):
            try:
                client = ollama.Client(
                    host=self._ollama_host,
                    timeout=settings.ollama_request_timeout_s,
                )
                kwargs: Dict[str, Any] = {
                    "model": settings.ollama_embed_model,
                    "input": text,
                }
                if settings.ollama_embed_dimensions is not None:
                    kwargs["dimensions"] = settings.ollama_embed_dimensions
                response = client.embed(**kwargs)
                vec = response.embeddings[0]
                return list(vec)
            except Exception as e:
                last_err = e
                log.warning(
                    "Ollama embed 失败 attempt=%s/%s: %s",
                    attempt + 1,
                    settings.ollama_embed_retries,
                    e,
                )
                if attempt + 1 < settings.ollama_embed_retries:
                    time.sleep(0.2 * (2**attempt) + random.random() * 0.1)
        raise RuntimeError(
            f"Ollama 嵌入在 {settings.ollama_embed_retries} 次重试后仍失败: {last_err}"
        ) from last_err

    def _get_embeddings_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            client = ollama.Client(
                host=self._ollama_host,
                timeout=settings.ollama_request_timeout_s,
            )
            kwargs: Dict[str, Any] = {
                "model": settings.ollama_embed_model,
                "input": list(texts),
            }
            if settings.ollama_embed_dimensions is not None:
                kwargs["dimensions"] = settings.ollama_embed_dimensions
            response = client.embed(**kwargs)
            return [list(v) for v in response.embeddings]
        except Exception as e:
            log.warning("Ollama 批量 embed 不可用，改为逐条: %s", e)
            return [self._get_embedding(t) for t in texts]

    def _init_milvus_collection(self) -> Collection:
        name = settings.milvus_collection
        dim = settings.vector_dim
        if utility.has_collection(name):
            col = Collection(name)
            col.load()
            return col

        fields = [
            FieldSchema(
                name="memory_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=True,
            ),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=32),
        ]
        schema = CollectionSchema(fields, description="v4 unified memory vectors")
        col = Collection(name, schema)
        col.create_index(
            "embedding",
            {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        for scalar in ("user_id", "timestamp", "memory_type"):
            try:
                col.create_index(scalar, {"index_type": "AUTOINDEX"})
            except Exception:
                pass
        col.load()
        return col

    def _ensure_neo4j_constraints(self) -> None:
        assert self._driver is not None
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE"
            )

    def _insert_milvus_v4(
        self,
        memory_ids: List[str],
        user_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        timestamps: List[int],
        memory_types: List[str],
    ) -> None:
        assert self._collection is not None
        n = len(memory_ids)
        if n == 0:
            return
        if not (
            n == len(user_ids) == len(texts) == len(embeddings) == len(timestamps) == len(memory_types)
        ):
            raise ValueError("Milvus v4 插入各列长度不一致")
        exp_dim = int(settings.vector_dim)
        for i, vec in enumerate(embeddings):
            if len(vec) != exp_dim:
                raise ValueError(
                    f"第 {i} 条嵌入维度为 {len(vec)}，与 vector_dim={exp_dim} 不一致；"
                    "请检查 OLLAMA_EMBED_MODEL / OLLAMA_EMBED_DIMENSIONS 与 Milvus 集合是否一致。"
                )
        try:
            self._collection.insert(
                [memory_ids, user_ids, texts, embeddings, timestamps, memory_types]
            )
        except Exception as e:
            raise RuntimeError(
                f"Milvus insert 失败 rows={n} memory_types_sample={memory_types[:3]!r}"
            ) from e

    @staticmethod
    def _node_element_id(node: Any) -> str:
        eid = getattr(node, "element_id", None)
        if eid is not None:
            return str(eid)
        return str(node.id)

    def _merge_memory_rel(
        self, session, source_element_id: str, target_element_id: str, rtype: str
    ) -> None:
        queries = {
            "NEXT": """
                MATCH (s) WHERE elementId(s) = $sid
                MATCH (t) WHERE elementId(t) = $tid
                MERGE (s)-[:NEXT]->(t)
            """,
            "CAUSED": """
                MATCH (s) WHERE elementId(s) = $sid
                MATCH (t) WHERE elementId(t) = $tid
                MERGE (s)-[:CAUSED]->(t)
            """,
            "SUB_EVENT_OF": """
                MATCH (s) WHERE elementId(s) = $sid
                MATCH (t) WHERE elementId(t) = $tid
                MERGE (s)-[:SUB_EVENT_OF]->(t)
            """,
            "RELATED": """
                MATCH (s) WHERE elementId(s) = $sid
                MATCH (t) WHERE elementId(t) = $tid
                MERGE (s)-[:RELATED]->(t)
            """,
        }
        q = queries.get(rtype)
        if q:
            session.run(q, sid=source_element_id, tid=target_element_id)

    def store_memory(
        self,
        user_id: str,
        memories: Optional[List[Dict[str, Any]]] = None,
        relations: Optional[List[Dict[str, Any]]] = None,
        # 以下参数已废弃，保留签名兼容以免旧调用崩溃（勿用于新代码）
        events: Any = None,
        facts: Any = None,
        knowledge: Any = None,
    ) -> Dict[str, Any]:
        assert self._driver is not None and self._collection is not None
        if events is not None or facts is not None or knowledge is not None:
            log.warning(
                "store_memory 收到已废弃的 events/facts/knowledge 参数，v4 请仅使用 memories"
            )
        memories = memories or []
        relations = relations or []
        log.info(
            "store_memory v5 user_id=%s memories=%s relations=%s",
            user_id,
            len(memories),
            len(relations),
        )

        memory_ids_out: List[str] = []
        milvus_mids: List[str] = []
        milvus_uids: List[str] = []
        milvus_texts: List[str] = []
        milvus_ts: List[int] = []
        milvus_types: List[str] = []

        temp_id_to_element_id: Dict[str, str] = {}

        with self._driver.session() as session:
            self._create_user(session, user_id)

            for idx, item in enumerate(memories):
                mtype = (item.get("type") or "").strip().lower()
                if mtype not in (MemoryType.event.value, MemoryType.fact.value, MemoryType.knowledge.value):
                    log.warning("跳过无效 memory type=%r", item.get("type"))
                    continue
                core = (item.get("content") or "").strip()
                if not core:
                    log.warning("跳过无 content 的记忆项 index=%s", idx)
                    continue

                time_disp, ts = _parse_time_string_to_ts(item.get("time"))
                loc = (item.get("location") or "").strip()
                subj = (item.get("subject") or "").strip()
                src = (item.get("source") or "").strip()
                tense_v = _normalize_optional_enum(item.get("tense"), TENSE_ALLOWED)
                conf_v = _normalize_optional_enum(
                    item.get("confidence"), CONFIDENCE_ALLOWED
                )

                full_content = _build_unified_memory_content(
                    mtype, time_disp, loc, subj, core, src
                )
                memory_id = f"mem_{self._id_gen.next_str()}"
                memory_ids_out.append(memory_id)

                fk: Optional[str] = None
                fv: Optional[str] = None
                if mtype == MemoryType.fact.value:
                    parsed = _parse_fact_kv_from_core(core)
                    if parsed:
                        fk, fv = parsed

                superseded_id: Optional[str] = None
                if mtype == MemoryType.fact.value and fk is not None and fv is not None:
                    old = session.run(
                        """
                        MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory)
                        WHERE m.memory_type = 'fact' AND m.fact_key = $fk
                          AND NOT (m)<-[:OVERRIDES]-(:Memory)
                        RETURN m ORDER BY m.timestamp DESC LIMIT 1
                        """,
                        user_id=user_id,
                        fk=fk,
                    ).single()
                    if (
                        old
                        and old["m"].get("fact_value") is not None
                        and str(old["m"].get("fact_value")) != fv
                    ):
                        superseded_id = old["m"]["memory_id"]

                row = session.run(
                    """
                    MATCH (u:User {user_id: $user_id})
                    CREATE (m:Memory {
                        memory_id: $memory_id,
                        content: $content,
                        timestamp: $ts,
                        memory_type: $mtype,
                        fact_key: $fk,
                        fact_value: $fv,
                        tense: $tense,
                        confidence: $confidence
                    })
                    CREATE (u)-[:HAS_MEMORY]->(m)
                    RETURN m
                    """,
                    user_id=user_id,
                    memory_id=memory_id,
                    content=full_content,
                    ts=ts,
                    mtype=mtype,
                    fk=fk,
                    fv=fv,
                    tense=tense_v,
                    confidence=conf_v,
                ).single()
                mnode = row["m"] if row else None

                tid = (item.get("temp_id") or item.get("client_ref") or str(idx)).strip()
                if mnode is not None:
                    temp_id_to_element_id[tid] = self._node_element_id(mnode)

                if superseded_id:
                    session.run(
                        """
                        MATCH (new:Memory {memory_id: $new_id})
                        MATCH (old:Memory {memory_id: $old_id})
                        MERGE (new)-[:OVERRIDES {timestamp: $ots}]->(old)
                        """,
                        new_id=memory_id,
                        old_id=superseded_id,
                        ots=int(time.time()),
                    )

                milvus_mids.append(memory_id)
                milvus_uids.append(user_id)
                milvus_texts.append(full_content[:4096])
                milvus_ts.append(ts)
                milvus_types.append(mtype)

            allowed_rel = {"NEXT", "CAUSED", "SUB_EVENT_OF", "RELATED"}
            for rel in relations:
                rtype = (rel.get("type") or "").strip()
                if rtype not in allowed_rel:
                    continue
                sid_key = (rel.get("source_temp_id") or "").strip()
                tid_key = (rel.get("target_temp_id") or "").strip()
                se = temp_id_to_element_id.get(sid_key)
                te = temp_id_to_element_id.get(tid_key)
                if not se or not te:
                    log.warning(
                        "relations 跳过：无法解析 temp_id source=%r target=%r",
                        sid_key,
                        tid_key,
                    )
                    continue
                self._merge_memory_rel(session, se, te, rtype)
                if rtype == "SUB_EVENT_OF":
                    chk = session.run(
                        """
                        MATCH (s) WHERE elementId(s) = $sid
                        MATCH (t) WHERE elementId(t) = $tid
                        RETURN coalesce(s.timestamp, 0) AS st, coalesce(t.timestamp, 0) AS tt
                        """,
                        sid=se,
                        tid=te,
                    ).single()
                    if chk and int(chk.get("st") or 0) > int(chk.get("tt") or 0):
                        log.warning(
                            "SUB_EVENT_OF: 子记忆时间戳晚于父记忆，请确认方向为子→父"
                        )

        if milvus_mids:
            vecs = self._get_embeddings_batch(milvus_texts)
            self._insert_milvus_v4(
                milvus_mids, milvus_uids, milvus_texts, vecs, milvus_ts, milvus_types
            )

        self._collection.flush()
        msg = f"已存储 {len(memory_ids_out)} 条记忆"
        log.info("store_memory v5 完成 %s", msg)
        return {"success": True, "memory_ids": memory_ids_out, "message": msg}

    def _create_user(self, session, user_id: str) -> None:
        session.run("MERGE (:User {user_id: $user_id})", user_id=user_id)

    def _current_fact_memory(
        self,
        session: Any,
        user_id: str,
        fact_key: str,
        filter_tense: Optional[str] = None,
        filter_confidence: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        rec = session.run(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory)
            WHERE m.memory_type = 'fact' AND m.fact_key = $fk
              AND NOT (m)<-[:OVERRIDES]-(:Memory)
              AND ($ft IS NULL OR m.tense = $ft)
              AND ($fc IS NULL OR m.confidence = $fc)
            RETURN m.memory_id AS memory_id, m.content AS content,
                   m.timestamp AS timestamp, m.memory_type AS memory_type,
                   m.tense AS tense, m.confidence AS confidence
            ORDER BY m.timestamp DESC
            LIMIT 1
            """,
            user_id=user_id,
            fk=fact_key,
            ft=filter_tense,
            fc=filter_confidence,
        ).single()
        return dict(rec) if rec else None

    def _parse_fact_key_from_stored_content(self, content: str) -> Optional[str]:
        """从已落库的完整模板中解析事实键（与 fact_key 属性互为补充）。"""
        for raw in (content or "").splitlines():
            line = raw.strip()
            if line.startswith("【内容】"):
                inner = line.replace("【内容】", "", 1).strip()
                if "=" in inner:
                    return inner.split("=", 1)[0].strip()
        return None

    def _causal_chain_for_memory_event(
        self, session: Any, memory_id: str
    ) -> List[str]:
        rec = session.run(
            """
            MATCH (e:Memory {memory_id: $mid})
            WHERE e.memory_type = 'event'
            OPTIONAL MATCH p = (e)<-[:CAUSED*1..15]-(r:Memory)
            WHERE r.memory_type = 'event' AND NOT (r)<-[:CAUSED]-(:Memory)
            WITH e, p
            ORDER BY length(p) DESC
            LIMIT 1
            RETURN e.content AS leaf_c,
                   CASE WHEN p IS NULL THEN []
                        ELSE [n IN reverse(nodes(p)) WHERE elementId(n) <> elementId(e) | n.content]
                   END AS chain
            """,
            mid=memory_id,
        ).single()
        if not rec:
            return []
        out: List[str] = []
        for c in rec.get("chain") or []:
            if c and str(c).strip():
                out.append(_content_line_summary(str(c)))
        leaf = rec.get("leaf_c")
        if leaf and str(leaf).strip():
            out.append(_content_line_summary(str(leaf)))
        return out

    def query_memory(
        self,
        user_id: str,
        query_text: str,
        memory_types: Optional[List[str]] = None,
        time_start: Optional[int] = None,
        time_end: Optional[int] = None,
        top_k: int = 5,
        tense: Optional[str] = None,
        confidence: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert self._driver is not None and self._collection is not None
        qt = (query_text or "").strip()
        if not qt:
            return {
                "memories": [],
                "causal_chain": [],
                "error": "query_text 不能为空",
            }

        ft = _normalize_optional_enum(tense, TENSE_ALLOWED)
        fc = _normalize_optional_enum(confidence, CONFIDENCE_ALLOWED)

        col = self._collection
        uid_expr = _milvus_escape(user_id)
        search_ef = settings.milvus_search_ef
        mem_types = [str(x).strip().lower() for x in (memory_types or []) if str(x).strip()]
        valid_t = {MemoryType.event.value, MemoryType.fact.value, MemoryType.knowledge.value}
        mem_types = [x for x in mem_types if x in valid_t]

        expr_parts = [f"user_id == '{uid_expr}'"]
        if mem_types:
            ors = " or ".join(f"memory_type == '{_milvus_escape(t)}'" for t in mem_types)
            expr_parts.append(f"({ors})")
        if time_start is not None:
            expr_parts.append(f"timestamp >= {int(time_start)}")
        if time_end is not None:
            expr_parts.append(f"timestamp <= {int(time_end)}")
        expr = " and ".join(expr_parts)

        search_limit = max(top_k * 2, top_k)
        if ft or fc:
            search_limit = max(top_k * 4, 32, top_k * 2)

        vec = self._get_embedding(qt[:2048])
        hits = col.search(
            data=[vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": search_ef}},
            limit=search_limit,
            expr=expr,
            output_fields=["memory_id", "memory_type"],
        )

        seen_mid: set[str] = set()
        ordered_mids: List[str] = []
        for hit in hits[0]:
            mid = hit.entity.get("memory_id")
            if not mid:
                continue
            sid = str(mid)
            if sid in seen_mid:
                continue
            seen_mid.add(sid)
            ordered_mids.append(sid)
            if len(ordered_mids) >= search_limit:
                break

        memories: List[Dict[str, Any]] = []
        causal_chain: List[str] = []

        with self._driver.session() as session:
            resolved_keys: set[str] = set()
            for mid in ordered_mids:
                if len(memories) >= top_k:
                    break
                rec = session.run(
                    """
                    MATCH (m:Memory {memory_id: $mid})
                    WHERE ($ft IS NULL OR m.tense = $ft)
                      AND ($fc IS NULL OR m.confidence = $fc)
                    RETURN m.content AS content, m.timestamp AS timestamp,
                           m.memory_type AS memory_type, m.fact_key AS fact_key,
                           m.tense AS tense, m.confidence AS confidence
                    """,
                    mid=mid,
                    ft=ft,
                    fc=fc,
                ).single()
                if not rec:
                    continue
                mtype = (rec.get("memory_type") or "").strip()
                fk = rec.get("fact_key")

                if mtype == MemoryType.fact.value:
                    key = fk or self._parse_fact_key_from_stored_content(
                        rec.get("content") or ""
                    )
                    if key:
                        if key in resolved_keys:
                            continue
                        resolved_keys.add(key)
                        cur = self._current_fact_memory(
                            session, user_id, key, ft, fc
                        )
                        if cur:
                            memories.append(
                                _memory_payload_from_record(cur, str(cur["memory_id"]))
                            )
                        continue
                    memories.append(_memory_payload_from_record(rec, mid))
                    continue

                memories.append(_memory_payload_from_record(rec, mid))

            if memories and memories[0].get("memory_type") == MemoryType.event.value:
                causal_chain = self._causal_chain_for_memory_event(
                    session, memories[0]["memory_id"]
                )

        out: Dict[str, Any] = {"memories": memories}
        if causal_chain:
            out["causal_chain"] = causal_chain
        log.info(
            "query_memory v5 完成 user_id=%s hits=%d memories_out=%d tense=%s confidence=%s",
            user_id,
            len(ordered_mids),
            len(memories),
            ft,
            fc,
        )
        return out
