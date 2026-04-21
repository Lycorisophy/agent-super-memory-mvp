"""
长期记忆存储与查询（Milvus + Neo4j），实现见 design.md。
"""
from __future__ import annotations

import logging
import random
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

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


class RefType(str, Enum):
    """Milvus / 查询中的 ref_type 与 source_type 常量。"""

    event = "event"
    fact = "fact"
    knowledge = "knowledge"
    structured = "structured"


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


def _milvus_fact_embed_text(key: str, value: str) -> str:
    """写入 Milvus 的向量化文本：固定中文句式，键与值完全使用存储时传入的字符串。"""
    k = (key or "").strip()
    v = (value or "").strip()
    return f"事实：键为「{k}」，值为「{v}」。"


def _fact_vector_probe_text(
    fact_keys: Optional[List[str]],
    global_vector_fallback: Optional[Dict[str, Any]],
) -> str:
    """事实向量检索用查询句：用户原话 + 模型在 fact_keys 里给的键名原文，不做代码侧改写。"""
    parts: List[str] = []
    if global_vector_fallback:
        t = (global_vector_fallback.get("text") or "").strip()
        if t:
            parts.append(t)
    if fact_keys:
        joined = "、".join(str(x).strip() for x in fact_keys if str(x).strip())
        if joined:
            parts.append("相关事实键：" + joined)
    out = "。".join(parts)
    return out if out else "个人事实与资料"


def _milvus_event_text_cn(
    summary: str,
    time_ts: int,
    location: str,
    subject: str,
    participants: List[str],
    action: str,
    result: str,
    tense: str,
    confidence: str,
    entities: List[str],
) -> str:
    pts = "、".join(participants or [])
    ents = "、".join(entities or [])
    parts = [
        f"事件摘要：{summary}。",
        f"发生时间为第{time_ts}秒（时间戳）。",
        f"地点：{location}。" if location else "",
        f"主体：{subject}。" if subject else "",
        f"参与人员：{pts}。" if pts else "",
        f"经过：{action}。" if action else "",
        f"结果：{result}。" if result else "",
        f"时态：{tense or '未填'}。",
        f"可信程度：{confidence or '未填'}。",
    ]
    if ents:
        parts.append(f"涉及人物或事物：{ents}。")
    return "".join(parts)


def _milvus_knowledge_text_cn(title: str, content: str, category: str) -> str:
    cat = category or "通用"
    return f"知识：《{title}》。要点：{content}。类别：{cat}。"


class MemorySystem:
    """连接 Milvus / Neo4j，提供 store_memory 与 query_memory。"""

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
        """单条文本嵌入；带超时与有限次重试。"""
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
        """多条文本一次请求嵌入；失败则逐条回退。"""
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
                name="chunk_id",
                dtype=DataType.VARCHAR,
                max_length=64,
                is_primary=True,
            ),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="ref_type", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="ref_id", dtype=DataType.VARCHAR, max_length=64),
        ]
        schema = CollectionSchema(fields, description="Long-term memory chunks")
        col = Collection(name, schema)
        col.create_index(
            "embedding",
            {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        for scalar in ("user_id", "timestamp", "ref_type"):
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
                "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT fact_id_unique IF NOT EXISTS FOR (f:Fact) REQUIRE f.fact_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT knowledge_id_unique IF NOT EXISTS FOR (k:Knowledge) REQUIRE k.knowledge_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT entity_name_type IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE"
            )

    def _insert_milvus_rows(
        self,
        chunk_ids: List[str],
        user_ids: List[str],
        texts: List[str],
        embeddings: List[List[float]],
        timestamps: List[int],
        source_types: List[str],
        ref_types: List[str],
        ref_ids: List[str],
    ) -> None:
        assert self._collection is not None
        n = len(chunk_ids)
        if n == 0:
            return
        if not (
            n == len(user_ids) == len(texts) == len(embeddings) == len(timestamps)
            == len(source_types) == len(ref_types) == len(ref_ids)
        ):
            raise ValueError("Milvus 插入各列长度不一致")
        try:
            self._collection.insert(
                [
                    chunk_ids,
                    user_ids,
                    texts,
                    embeddings,
                    timestamps,
                    source_types,
                    ref_types,
                    ref_ids,
                ]
            )
        except Exception as e:
            raise RuntimeError(
                f"Milvus insert 失败 rows={n} ref_types_sample={ref_types[:3]!r}"
            ) from e

    def _create_user(self, session, user_id: str) -> None:
        session.run("MERGE (:User {user_id: $user_id})", user_id=user_id)

    def _resolve_node(
        self,
        session,
        user_id: str,
        summary: str,
        node_kind: str,
        *,
        temp_id_map: Optional[Dict[str, str]] = None,
        batch_event_nodes: Optional[List[Any]] = None,
        batch_fact_nodes: Optional[List[Any]] = None,
        batch_knowledge_nodes: Optional[List[Any]] = None,
    ) -> Optional[Any]:
        summary = (summary or "").strip()
        if not summary:
            return None
        if temp_id_map and summary in temp_id_map:
            eid = temp_id_map[summary]
            rec = session.run(
                "MATCH (n) WHERE elementId(n) = $eid RETURN n AS n",
                eid=eid,
            ).single()
            return rec["n"] if rec else None

        if node_kind == "event":
            if batch_event_nodes:
                for enode in batch_event_nodes:
                    s = (enode.get("summary") or "").strip()
                    a = (enode.get("action") or "").strip()
                    if summary == s or summary == a:
                        return enode
            rec = session.run(
                """
                MATCH (u:User {user_id: $user_id})-[:HAS_EVENT]->(e:Event)
                WHERE e.summary = $s OR e.action = $s
                RETURN e ORDER BY e.time DESC LIMIT 1
                """,
                user_id=user_id,
                s=summary,
            ).single()
            if rec:
                return rec["e"]
            rec = session.run(
                """
                MATCH (u:User {user_id: $user_id})-[:HAS_EVENT]->(e:Event)
                WHERE e.summary CONTAINS $s OR e.action CONTAINS $s
                RETURN e ORDER BY e.time DESC LIMIT 1
                """,
                user_id=user_id,
                s=summary,
            ).single()
            return rec["e"] if rec else None

        if node_kind == "fact":
            if batch_fact_nodes:
                for fnode in batch_fact_nodes:
                    k = (fnode.get("key") or "").strip()
                    if k == summary:
                        return fnode
            rec = session.run(
                """
                MATCH (u:User {user_id: $user_id})-[:HAS_FACT]->(f:Fact {key: $k})
                RETURN f ORDER BY f.updated_at DESC LIMIT 1
                """,
                user_id=user_id,
                k=summary,
            ).single()
            return rec["f"] if rec else None

        if node_kind == "knowledge":
            if summary.startswith("kn_"):
                rec = session.run(
                    """
                    MATCH (k:Knowledge {knowledge_id: $kid})
                    RETURN k LIMIT 1
                    """,
                    kid=summary,
                ).single()
                if rec:
                    return rec["k"]
            if batch_knowledge_nodes:
                for knode in batch_knowledge_nodes:
                    t = (knode.get("title") or "").strip()
                    if t == summary:
                        return knode
            rec = session.run(
                """
                MATCH (k:Knowledge {title: $t})
                RETURN k LIMIT 1
                """,
                t=summary,
            ).single()
            return rec["k"] if rec else None

        if node_kind == "entity":
            rec = session.run(
                """
                MATCH (ent:Entity)
                WHERE ent.name = $n
                RETURN ent LIMIT 1
                """,
                n=summary,
            ).single()
            return rec["ent"] if rec else None
        return None

    @staticmethod
    def _node_element_id(node: Any) -> str:
        eid = getattr(node, "element_id", None)
        if eid is not None:
            return str(eid)
        return str(node.id)

    def _merge_rel(
        self,
        session,
        source_element_id: str,
        target_element_id: str,
        rtype: str,
        overrides_ts: Optional[int] = None,
    ) -> None:
        if rtype == "OVERRIDES":
            ts = overrides_ts if overrides_ts is not None else int(time.time())
            session.run(
                """
                MATCH (s) WHERE elementId(s) = $sid
                MATCH (t) WHERE elementId(t) = $tid
                MERGE (s)-[r:OVERRIDES]->(t)
                SET r.timestamp = $ts
                """,
                sid=source_element_id,
                tid=target_element_id,
                ts=ts,
            )
            return
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
            "RELATED_TO": """
                MATCH (s) WHERE elementId(s) = $sid
                MATCH (t) WHERE elementId(t) = $tid
                MERGE (s)-[:RELATED_TO]->(t)
            """,
            "MENTIONS": """
                MATCH (s) WHERE elementId(s) = $sid
                MATCH (t) WHERE elementId(t) = $tid
                MERGE (s)-[:MENTIONS]->(t)
            """,
        }
        q = queries.get(rtype)
        if q:
            session.run(q, sid=source_element_id, tid=target_element_id)

    def store_memory(
        self,
        user_id: str,
        events: Optional[List[Dict[str, Any]]] = None,
        facts: Optional[List[Dict[str, Any]]] = None,
        knowledge: Optional[List[Dict[str, Any]]] = None,
        relations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        assert self._driver is not None and self._collection is not None
        log.info(
            "store_memory 开始 user_id=%s events=%s facts=%s knowledge=%s relations=%s",
            user_id,
            len(events or []),
            len(facts or []),
            len(knowledge or []),
            len(relations or []),
        )
        stored_ids: Dict[str, List[str]] = {
            "event_ids": [],
            "fact_ids": [],
            "knowledge_ids": [],
        }
        milvus_chunk_ids: List[str] = []
        milvus_user_ids: List[str] = []
        milvus_texts: List[str] = []
        milvus_timestamps: List[int] = []
        milvus_source_types: List[str] = []
        milvus_ref_types: List[str] = []
        milvus_ref_ids: List[str] = []
        temp_id_map: Dict[str, str] = {}
        batch_event_nodes: List[Any] = []
        batch_fact_nodes: List[Any] = []
        batch_knowledge_nodes: List[Any] = []

        with self._driver.session() as session:
            self._create_user(session, user_id)

            if events:
                for ev in events:
                    event_id = f"ev_{self._id_gen.next_str()}"
                    summary = ev.get("summary", "")
                    time_ts = int(ev.get("time", ev.get("timestamp", int(time.time()))))
                    location = ev.get("location", "") or ""
                    subject = ev.get("subject", "") or ""
                    participants = list(ev.get("participants", []) or [])
                    action = ev.get("action", "") or summary
                    result = ev.get("result", "") or ""
                    tense = ev.get("tense", "past")
                    confidence = ev.get("confidence", "real")

                    row_e = session.run(
                        """
                        MATCH (u:User {user_id: $user_id})
                        CREATE (e:Event {
                            event_id: $event_id, summary: $summary, time: $time,
                            location: $location, subject: $subject, participants: $participants,
                            action: $action, result: $result, tense: $tense, confidence: $confidence
                        })
                        CREATE (u)-[:HAS_EVENT]->(e)
                        RETURN e
                        """,
                        user_id=user_id,
                        event_id=event_id,
                        summary=summary,
                        time=time_ts,
                        location=location,
                        subject=subject,
                        participants=participants,
                        action=action,
                        result=result,
                        tense=tense,
                        confidence=confidence,
                    ).single()
                    enode = row_e["e"] if row_e else None
                    if enode is not None:
                        batch_event_nodes.append(enode)
                    tid = ev.get("temp_id") or ev.get("client_ref")
                    if tid and str(tid).strip() and enode is not None:
                        temp_id_map[str(tid).strip()] = self._node_element_id(enode)
                    stored_ids["event_ids"].append(event_id)

                    for ent in ev.get("entities", []) or []:
                        if not str(ent).strip():
                            continue
                        session.run(
                            """
                            MATCH (e:Event {event_id: $event_id})
                            MERGE (ent:Entity {name: $name, type: 'other'})
                            MERGE (e)-[:MENTIONS]->(ent)
                            """,
                            event_id=event_id,
                            name=str(ent).strip(),
                        )

                    embed_text = _milvus_event_text_cn(
                        summary=summary,
                        time_ts=time_ts,
                        location=location,
                        subject=subject,
                        participants=participants,
                        action=action,
                        result=result,
                        tense=tense,
                        confidence=confidence,
                        entities=list(ev.get("entities") or []),
                    )
                    milvus_chunk_ids.append(f"chk_{self._id_gen.next_str()}")
                    milvus_user_ids.append(user_id)
                    milvus_texts.append(embed_text[:4096])
                    milvus_timestamps.append(time_ts)
                    milvus_source_types.append(RefType.structured.value)
                    milvus_ref_types.append(RefType.event.value)
                    milvus_ref_ids.append(event_id)

            if facts:
                for fact in facts:
                    key, value = fact["key"], fact["value"]
                    confidence = fact.get("confidence", "high")
                    fact_id = f"fact_{self._id_gen.next_str()}"
                    now_ts = int(time.time())

                    old = session.run(
                        """
                        MATCH (u:User {user_id: $user_id})-[:HAS_FACT]->(f:Fact {key: $key})
                        RETURN f ORDER BY f.updated_at DESC LIMIT 1
                        """,
                        user_id=user_id,
                        key=key,
                    ).single()

                    row_f = session.run(
                        """
                        MATCH (u:User {user_id: $user_id})
                        CREATE (f:Fact {
                            fact_id: $fact_id, key: $key, value: $value,
                            updated_at: $ts, confidence: $confidence
                        })
                        CREATE (u)-[:HAS_FACT]->(f)
                        RETURN f
                        """,
                        user_id=user_id,
                        fact_id=fact_id,
                        key=key,
                        value=value,
                        ts=now_ts,
                        confidence=confidence,
                    ).single()
                    fnode = row_f["f"] if row_f else None
                    if fnode is not None:
                        batch_fact_nodes.append(fnode)
                    ftid = fact.get("temp_id") or fact.get("client_ref")
                    if ftid and str(ftid).strip() and fnode is not None:
                        temp_id_map[str(ftid).strip()] = self._node_element_id(fnode)
                    stored_ids["fact_ids"].append(fact_id)

                    if old and old["f"].get("value") != value:
                        session.run(
                            """
                            MATCH (new:Fact {fact_id: $new_id})
                            MATCH (old:Fact {fact_id: $old_id})
                            MERGE (new)-[:OVERRIDES {timestamp: $ts}]->(old)
                            """,
                            new_id=fact_id,
                            old_id=old["f"]["fact_id"],
                            ts=now_ts,
                        )

                    embed_text = _milvus_fact_embed_text(str(key), str(value))
                    milvus_chunk_ids.append(f"chk_{self._id_gen.next_str()}")
                    milvus_user_ids.append(user_id)
                    milvus_texts.append(embed_text[:4096])
                    milvus_timestamps.append(now_ts)
                    milvus_source_types.append(RefType.structured.value)
                    milvus_ref_types.append(RefType.fact.value)
                    milvus_ref_ids.append(fact_id)

            if knowledge:
                for kn in knowledge:
                    kn_id = f"kn_{self._id_gen.next_str()}"
                    title = kn["title"]
                    content = kn["content"]
                    category = kn.get("category", "general") or "general"
                    row_k = session.run(
                        """
                        CREATE (k:Knowledge {
                            knowledge_id: $kn_id, title: $title, content: $content, category: $category
                        })
                        RETURN k
                        """,
                        kn_id=kn_id,
                        title=title,
                        content=content,
                        category=category,
                    ).single()
                    knode = row_k["k"] if row_k else None
                    if knode is not None:
                        batch_knowledge_nodes.append(knode)
                    ktid = kn.get("temp_id") or kn.get("client_ref")
                    if ktid and str(ktid).strip() and knode is not None:
                        temp_id_map[str(ktid).strip()] = self._node_element_id(knode)
                    stored_ids["knowledge_ids"].append(kn_id)

                    embed_text = _milvus_knowledge_text_cn(
                        str(title), str(content), str(category)
                    )
                    ts = int(time.time())
                    milvus_chunk_ids.append(f"chk_{self._id_gen.next_str()}")
                    milvus_user_ids.append(user_id)
                    milvus_texts.append(embed_text[:4096])
                    milvus_timestamps.append(ts)
                    milvus_source_types.append(RefType.structured.value)
                    milvus_ref_types.append(RefType.knowledge.value)
                    milvus_ref_ids.append(kn_id)

            if relations:
                allowed = {
                    "NEXT",
                    "CAUSED",
                    "SUB_EVENT_OF",
                    "MENTIONS",
                    "OVERRIDES",
                    "RELATED_TO",
                }
                for rel in relations:
                    rtype = rel.get("type")
                    if rtype not in allowed:
                        continue
                    st = rel.get("source_type")
                    tt = rel.get("target_type")
                    src = self._resolve_node(
                        session,
                        user_id,
                        rel.get("source_summary", ""),
                        st or "event",
                        temp_id_map=temp_id_map,
                        batch_event_nodes=batch_event_nodes,
                        batch_fact_nodes=batch_fact_nodes,
                        batch_knowledge_nodes=batch_knowledge_nodes,
                    )
                    tgt = self._resolve_node(
                        session,
                        user_id,
                        rel.get("target_summary", ""),
                        tt or "event",
                        temp_id_map=temp_id_map,
                        batch_event_nodes=batch_event_nodes,
                        batch_fact_nodes=batch_fact_nodes,
                        batch_knowledge_nodes=batch_knowledge_nodes,
                    )
                    if src is None or tgt is None:
                        continue
                    sid = self._node_element_id(src)
                    tid = self._node_element_id(tgt)
                    self._merge_rel(
                        session,
                        sid,
                        tid,
                        rtype,
                        overrides_ts=int(time.time()) if rtype == "OVERRIDES" else None,
                    )
                    if rtype == "SUB_EVENT_OF":
                        chk = session.run(
                            """
                            MATCH (s) WHERE elementId(s) = $sid
                            MATCH (t) WHERE elementId(t) = $tid
                            RETURN coalesce(s.time, 0) AS st, coalesce(t.time, 0) AS tt
                            """,
                            sid=sid,
                            tid=tid,
                        ).single()
                        if chk and chk.get("st") is not None and chk.get("tt") is not None:
                            if int(chk["st"]) > int(chk["tt"]):
                                log.warning(
                                    "SUB_EVENT_OF: 子事件时间晚于父事件 (子 time=%s 父 time=%s)，请确认方向为子→父",
                                    chk["st"],
                                    chk["tt"],
                                )

        if milvus_texts:
            embeddings = self._get_embeddings_batch(milvus_texts)
            self._insert_milvus_rows(
                milvus_chunk_ids,
                milvus_user_ids,
                milvus_texts,
                embeddings,
                milvus_timestamps,
                milvus_source_types,
                milvus_ref_types,
                milvus_ref_ids,
            )

        log.info("store_memory: Milvus flush")
        self._collection.flush()
        msg = (
            f"已存储 {len(stored_ids['event_ids'])} 个事件, "
            f"{len(stored_ids['fact_ids'])} 个事实, "
            f"{len(stored_ids['knowledge_ids'])} 条知识"
        )
        log.info("store_memory 完成 %s", msg)
        return {"success": True, "stored_ids": stored_ids, "message": msg}

    def _causal_chain_for_event(self, session, event_id: str) -> List[str]:
        """自根因事件到当前事件的一条最长 CAUSED 链（单条 Cypher）。"""
        rec = session.run(
            """
            MATCH (e:Event {event_id: $eid})
            OPTIONAL MATCH p = (e)<-[:CAUSED*1..15]-(r:Event)
            WHERE NOT (r)<-[:CAUSED]-(:Event)
            WITH e, p
            ORDER BY length(p) DESC
            LIMIT 1
            RETURN e.summary AS leaf_s,
                   CASE WHEN p IS NULL THEN []
                        ELSE [n IN reverse(nodes(p)) WHERE elementId(n) <> elementId(e) | n.summary]
                   END AS chain
            """,
            eid=event_id,
        ).single()
        if not rec:
            return []
        out: List[str] = []
        for s in rec.get("chain") or []:
            if s and str(s).strip():
                out.append(str(s).strip())
        leaf = rec.get("leaf_s")
        if leaf and str(leaf).strip():
            out.append(str(leaf).strip())
        return out

    def _parent_summaries(self, session, event_id: str) -> List[str]:
        rows = session.run(
            """
            MATCH (e:Event {event_id: $eid})-[:SUB_EVENT_OF]->(p:Event)
            RETURN p.summary AS s
            """,
            eid=event_id,
        )
        return [r["s"] for r in rows if r.get("s")]

    def _fact_current_value(
        self, session: Any, user_id: str, key: str
    ) -> Optional[str]:
        """用户维度下某 key 的当前头事实（无 OVERRIDES 入边），按 updated_at 最新。"""
        rec = session.run(
            """
            MATCH (u:User {user_id: $user_id})-[:HAS_FACT]->(f:Fact {key: $key})
            WHERE NOT (f)<-[:OVERRIDES]-(:Fact)
            RETURN f.value AS value
            ORDER BY f.updated_at DESC
            LIMIT 1
            """,
            user_id=user_id,
            key=key,
        ).single()
        if not rec or rec.get("value") is None:
            return None
        return str(rec["value"])

    def _hydrate_facts_from_ref_ids(
        self,
        session: Any,
        result: Dict[str, Any],
        user_id: str,
        fact_ids: Sequence[str],
    ) -> None:
        """向量命中的多条 fact_id：先取 key，再按 key 回填用户下当前有效值。"""
        ordered: List[str] = []
        seen_f: set[str] = set()
        for fid in fact_ids:
            fs = str(fid).strip()
            if not fs or fs in seen_f:
                continue
            seen_f.add(fs)
            ordered.append(fs)
        if not ordered:
            return
        rows = session.run(
            """
            UNWIND $ids AS fid
            MATCH (f:Fact {fact_id: fid})
            RETURN f.fact_id AS fact_id, f.key AS key
            """,
            ids=ordered,
        )
        by_fid: Dict[str, str] = {}
        for r in rows:
            fid = r.get("fact_id")
            k = r.get("key")
            if fid is not None and k is not None:
                by_fid[str(fid)] = str(k)
        key_order: List[str] = []
        seen_keys: set[str] = set()
        for fid in ordered:
            k = by_fid.get(fid)
            if k is None:
                continue
            if k not in seen_keys:
                seen_keys.add(k)
                key_order.append(k)
        for key in key_order:
            val = self._fact_current_value(session, user_id, key)
            if val is not None:
                result["facts"][key] = val

    def query_memory(
        self,
        user_id: str,
        fact_keys: Optional[List[str]] = None,
        event_query: Optional[Dict[str, Any]] = None,
        knowledge_query: Optional[Dict[str, Any]] = None,
        global_vector_fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert self._driver is not None and self._collection is not None
        log.info(
            "query_memory 开始 user_id=%s fact_keys=%s has_event_query=%s has_knowledge_query=%s has_fallback=%s",
            user_id,
            fact_keys,
            bool(event_query),
            bool(knowledge_query),
            bool(global_vector_fallback),
        )
        col = self._collection
        uid_expr = _milvus_escape(user_id)
        search_ef = settings.milvus_search_ef
        result: Dict[str, Any] = {
            "facts": {},
            "events": [],
            "knowledge": [],
            "vector_chunks": [],
            "causal_chains": [],
            "parent_events": [],
        }

        with self._driver.session() as session:
            # 事实检索：以 Milvus 向量（ref_type=fact）为主，中文查询句；Neo4j 仅按命中 fact_id 回填 facts
            if fact_keys or (
                global_vector_fallback
                and (global_vector_fallback.get("text") or "").strip()
            ):
                probe = _fact_vector_probe_text(fact_keys, global_vector_fallback)
                fact_top_k = 12
                log.info(
                    "query_memory: Milvus 事实向量检索（主路径）top_k=%s probe_len=%s",
                    fact_top_k,
                    len(probe),
                )
                log.debug("query_memory: 事实 probe 预览=%r", probe[:240])
                vec_facts = self._get_embedding(probe[:2048])
                expr_fact = (
                    f"user_id == '{uid_expr}' and ref_type == '{RefType.fact.value}'"
                )
                hits_fact = col.search(
                    data=[vec_facts],
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "params": {"ef": search_ef}},
                    limit=fact_top_k,
                    expr=expr_fact,
                    output_fields=["text", "ref_id"],
                )
                log.info(
                    "query_memory: Milvus 事实向量原始命中数=%d",
                    len(hits_fact[0]),
                )
                fact_ids_hit: List[str] = []
                for i, hit in enumerate(hits_fact[0]):
                    t = hit.entity.get("text")
                    fid = hit.entity.get("ref_id")
                    log.debug(
                        "  fact_vec[%d] distance=%s ref_id=%s text_preview=%r",
                        i,
                        getattr(hit, "distance", None),
                        fid,
                        (t or "")[:120],
                    )
                    if t and t not in result["vector_chunks"]:
                        result["vector_chunks"].append(t)
                    if fid:
                        fact_ids_hit.append(str(fid))
                self._hydrate_facts_from_ref_ids(
                    session, result, user_id, fact_ids_hit
                )
                log.info(
                    "query_memory: 事实向量阶段结束 facts 条数=%d vector_chunks 条数=%d",
                    len(result["facts"]),
                    len(result["vector_chunks"]),
                )

            if event_query:
                sem_text = event_query.get("semantic_text") or ""
                t_start = event_query.get("time_start")
                t_end = event_query.get("time_end")
                entities = list(event_query.get("entities") or [])
                top_k = int(event_query.get("top_k", 3))

                vec = self._get_embedding(sem_text) if sem_text else self._get_embedding(
                    "事件"
                )
                expr_parts = [
                    f"user_id == '{uid_expr}'",
                    f"ref_type == '{RefType.event.value}'",
                ]
                if t_start is not None:
                    expr_parts.append(f"timestamp >= {int(t_start)}")
                if t_end is not None:
                    expr_parts.append(f"timestamp <= {int(t_end)}")
                expr = " and ".join(expr_parts)
                log.info(
                    "query_memory: Milvus 事件向量检索 top_k=%s expr=%s",
                    top_k,
                    expr[:200],
                )

                hits = col.search(
                    data=[vec],
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "params": {"ef": search_ef}},
                    limit=max(top_k * 2, top_k),
                    expr=expr,
                    output_fields=["ref_id"],
                )
                event_ids: List[str] = []
                for hit in hits[0]:
                    rid = hit.entity.get("ref_id")
                    if rid and rid not in event_ids:
                        event_ids.append(rid)

                log.info(
                    "query_memory: Milvus 事件向量原始命中数=%d 去重后 event_ids=%s",
                    len(hits[0]),
                    event_ids[:12],
                )
                for i, hit in enumerate(hits[0][:8]):
                    log.debug(
                        "  event_vec[%d] distance=%s ref_id=%s",
                        i,
                        getattr(hit, "distance", None),
                        hit.entity.get("ref_id"),
                    )

                added = 0
                for eid in event_ids:
                    if added >= top_k:
                        break
                    ev = session.run(
                        """
                        MATCH (e:Event {event_id: $eid})
                        OPTIONAL MATCH (e)-[:MENTIONS]->(ent:Entity)
                        RETURN e.summary AS summary, e.time AS time, e.location AS location,
                               e.subject AS subject, e.participants AS participants,
                               e.action AS action, e.result AS result, e.tense AS tense,
                               e.confidence AS confidence, collect(DISTINCT ent.name) AS entities
                        """,
                        eid=eid,
                    ).single()
                    if not ev:
                        continue
                    ent_list = [x for x in (ev["entities"] or []) if x]
                    if entities and not any(e in ent_list for e in entities):
                        continue
                    row = {
                        "summary": ev["summary"],
                        "time": ev["time"],
                        "location": ev["location"],
                        "subject": ev["subject"],
                        "participants": ev["participants"] or [],
                        "action": ev["action"],
                        "result": ev["result"],
                        "tense": ev["tense"],
                        "confidence": ev["confidence"],
                        "entities": ent_list,
                        "parents": self._parent_summaries(session, eid),
                    }
                    result["events"].append(row)
                    chain = self._causal_chain_for_event(session, eid)
                    if len(chain) > 1:
                        result["causal_chains"].append(chain)
                    parents = row.get("parents") or []
                    for p in parents:
                        if p and p not in result["parent_events"]:
                            result["parent_events"].append(p)
                    added += 1

                log.info(
                    "query_memory: 事件经 Neo4j 装配后写入条数=%d (Milvus 候选=%d, entities 过滤=%s)",
                    len(result["events"]),
                    len(event_ids),
                    bool(entities),
                )

            if knowledge_query:
                sem_text = knowledge_query.get("semantic_text") or ""
                category = knowledge_query.get("category")
                top_k = int(knowledge_query.get("top_k", 2))
                vec = self._get_embedding(sem_text) if sem_text else self._get_embedding(
                    "知识"
                )
                log.info(
                    "query_memory: Milvus 知识向量检索 top_k=%s category=%r",
                    top_k,
                    category,
                )
                hits = col.search(
                    data=[vec],
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "params": {"ef": search_ef}},
                    limit=max(top_k * 4, top_k),
                    expr=(
                        f"user_id == '{uid_expr}' and ref_type == '{RefType.knowledge.value}'"
                    ),
                    output_fields=["ref_id"],
                )
                for hit in hits[0]:
                    if len(result["knowledge"]) >= top_k:
                        break
                    kid = hit.entity.get("ref_id")
                    kn = session.run(
                        """
                        MATCH (k:Knowledge {knowledge_id: $kid})
                        RETURN k.title AS title, k.content AS content, k.category AS category
                        """,
                        kid=kid,
                    ).single()
                    if not kn:
                        continue
                    if category and kn.get("category") != category:
                        continue
                    result["knowledge"].append(
                        {"title": kn["title"], "content": kn["content"]}
                    )

        has_structured = bool(
            result["facts"] or result["events"] or result["knowledge"]
        )
        fact_requested_but_empty = bool(fact_keys) and not result["facts"]
        if global_vector_fallback:
            text = (global_vector_fallback.get("text") or "").strip()
            if text:
                run_full_fallback = (not has_structured) or fact_requested_but_empty
                if run_full_fallback:
                    top_k = int(global_vector_fallback.get("top_k", 5))
                    log.info(
                        "query_memory: Milvus 全文兜底 触发 "
                        "(无任何结构化=%s, 请求了fact_keys但facts仍空=%s) top_k=%s text_len=%s",
                        not has_structured,
                        fact_requested_but_empty,
                        top_k,
                        len(text),
                    )
                    vec = self._get_embedding(text)
                    hits = col.search(
                        data=[vec],
                        anns_field="embedding",
                        param={"metric_type": "COSINE", "params": {"ef": search_ef}},
                        limit=top_k,
                        expr=f"user_id == '{uid_expr}'",
                        output_fields=["text", "ref_type", "ref_id"],
                    )
                    fallback_fact_ids: List[str] = []
                    for i, hit in enumerate(hits[0]):
                        t = hit.entity.get("text")
                        rt = hit.entity.get("ref_type")
                        rid = hit.entity.get("ref_id")
                        log.debug(
                            "query_memory: 全文兜底 hit[%d] distance=%s ref_type=%s ref_id=%s text_preview=%r",
                            i,
                            getattr(hit, "distance", None),
                            rt,
                            rid,
                            (t or "")[:100],
                        )
                        if t and t not in result["vector_chunks"]:
                            result["vector_chunks"].append(t)
                        if rt == RefType.fact.value and rid:
                            fallback_fact_ids.append(str(rid))
                    if fallback_fact_ids and self._driver is not None:
                        with self._driver.session() as fb_session:
                            self._hydrate_facts_from_ref_ids(
                                fb_session, result, user_id, fallback_fact_ids
                            )

        if not result["causal_chains"]:
            del result["causal_chains"]
        if not result["parent_events"]:
            del result["parent_events"]

        log.info(
            "query_memory 完成 facts=%d events=%d knowledge=%d vector_chunks=%d",
            len(result.get("facts") or {}),
            len(result.get("events") or []),
            len(result.get("knowledge") or []),
            len(result.get("vector_chunks") or []),
        )
        return result
