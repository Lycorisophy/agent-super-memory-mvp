"""
表结构知识库：Markdown / Excel / 自然语言（LLM）导入 Milvus + Neo4j。
支持统一表 JSON、MD/Excel 键列与「关联」列解析、RELATES_TO 边写入。
Column 使用 column_key + 单列 UNIQUE（兼容 Neo4j 社区版）。
"""
from __future__ import annotations

import json
import logging
import re
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import ollama
import pandas as pd
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility
from sqlalchemy import create_engine, text

from config import settings
from memory_service import MemorySystem, _milvus_collection_embedding_dim

log = logging.getLogger(__name__)

SCHEMA_COLLECTION_NAME = "schema_docs"

# 整句扫描：防 SELECT ...; DROP ... 等拼接
_FORBIDDEN_SQL = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|TRUNCATE|GRANT|REVOKE|CALL)\b",
    re.IGNORECASE,
)


def is_safe_sql(sql: str) -> bool:
    """仅允许只读类语句，且禁止危险关键字与多语句。"""
    s = (sql or "").strip()
    if not s:
        return False
    upper = s.upper()
    allowed_starts = ("SELECT", "SHOW", "DESCRIBE", "DESC", "EXPLAIN", "WITH")
    ok_prefix = False
    for w in allowed_starts:
        if upper == w or upper.startswith(w + " ") or upper.startswith(w + "\t") or upper.startswith(w + "("):
            ok_prefix = True
            break
    if not ok_prefix:
        return False
    if _FORBIDDEN_SQL.search(s):
        return False
    core = s.rstrip().rstrip(";").strip()
    if ";" in core:
        return False
    return True


def parse_sql_explanation_from_llm(text: str) -> Tuple[str, str]:
    """解析 LLM 输出中的 SQL 与说明（供单测）。"""
    raw = (text or "").strip()
    sql = ""
    expl = ""
    m = re.search(r"```sql\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if m:
        sql = m.group(1).strip()
    for line in raw.splitlines():
        line = line.strip()
        if re.match(r"(?i)^sql\s*:", line):
            sql = re.sub(r"(?i)^sql\s*:", "", line).strip()
        elif re.match(r"(?i)^explanation\s*:", line):
            expl = re.sub(r"(?i)^explanation\s*:", "", line).strip()
    if sql:
        sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"\s*```$", "", sql).strip()
    return sql, expl


def normalize_schema_identifier(name: str) -> str:
    """表名/字段名：小写、空白与连字符转下划线，仅保留 [a-z0-9_]。"""
    s = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def _strip_json_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```\w*\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s).strip()
    return s


def _finalize_table_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """补全缺省字段并规范化标识符。"""
    name = normalize_schema_identifier(str(raw.get("name", "")))
    if not name:
        return {}
    cols_in = raw.get("columns") if isinstance(raw.get("columns"), list) else []
    columns: List[Dict[str, Any]] = []
    for c in cols_in:
        if not isinstance(c, dict):
            continue
        cn = normalize_schema_identifier(str(c.get("name", "")))
        if not cn:
            continue
        key = str(c.get("key", "") or "").strip().upper()
        if key not in ("PRI", "UNI", "MUL", ""):
            key = ""
        columns.append(
            {
                "name": cn,
                "type": str(c.get("type", "VARCHAR(255)") or "VARCHAR(255)").strip(),
                "nullable": bool(c.get("nullable", True)),
                "key": key,
                "default": c.get("default"),
                "comment": str(c.get("comment", "") or ""),
            }
        )
    rels_in = raw.get("relations") if isinstance(raw.get("relations"), list) else []
    relations: List[Dict[str, Any]] = []
    for r in rels_in:
        if not isinstance(r, dict):
            continue
        tt = normalize_schema_identifier(str(r.get("target_table", "")))
        if not tt:
            continue
        st = str(r.get("type", "belongs_to") or "belongs_to").strip().lower()
        sc = [normalize_schema_identifier(str(x)) for x in (r.get("source_cols") or []) if str(x).strip()]
        tc = [normalize_schema_identifier(str(x)) for x in (r.get("target_cols") or []) if str(x).strip()]
        if not sc or not tc:
            continue
        relations.append({"type": st, "target_table": tt, "source_cols": sc, "target_cols": tc})
    return {"name": name, "comment": str(raw.get("comment", "") or ""), "columns": columns, "relations": relations}


def _finalize_tables_list(items: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            t = _finalize_table_dict(item)
            if t.get("name") and t.get("columns"):
                out.append(t)
    return out


_LINK_CELL = re.compile(r"^\s*([\w]+)\.([\w]+)\s*$", re.IGNORECASE)


def _relation_from_link_cell(source_col: str, link_cell: str) -> Optional[Dict[str, Any]]:
    m = _LINK_CELL.match((link_cell or "").strip())
    if not m:
        return None
    tgt_table = normalize_schema_identifier(m.group(1))
    tgt_col = normalize_schema_identifier(m.group(2))
    sc = normalize_schema_identifier(source_col)
    if not tgt_table or not tgt_col or not sc:
        return None
    return {
        "type": "belongs_to",
        "target_table": tgt_table,
        "source_cols": [sc],
        "target_cols": [tgt_col],
    }


def _markdown_header_indices(header_cells: List[str]) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for i, raw in enumerate(header_cells):
        c = raw.strip().lower()
        if c in ("字段名", "field", "column", "列名", "name"):
            idx["name"] = i
        elif c in ("类型", "type"):
            idx["type"] = i
        elif c in ("注释", "comment", "说明", "描述", "description"):
            idx["comment"] = i
        elif c in ("键", "key", "索引", "pk"):
            idx["key"] = i
        elif c in ("关联", "foreign", "relation", "外键", "关联关系", "fk"):
            idx["link"] = i
    return idx


def _is_md_separator_row(parts: List[str]) -> bool:
    if not parts:
        return False
    return any("---" in p or ":--" in p for p in parts)


def _dedupe_relations(rels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: List[Dict[str, Any]] = []
    for r in rels:
        k = (r.get("type"), r.get("target_table"), tuple(r.get("source_cols") or []), tuple(r.get("target_cols") or []))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


NL_TABLES_PROMPT = """你是一个数据库建模专家。请根据用户的自然语言描述，提取其中涉及的数据表结构。

输出必须是一个严格的 JSON 数组，每个元素代表一张表，格式如下：

[
  {{
    "name": "表名（英文 snake_case）",
    "comment": "表的中文注释",
    "columns": [
      {{
        "name": "字段名",
        "type": "MySQL 数据类型",
        "nullable": true,
        "key": "PRI 或 UNI 或 MUL 或空字符串",
        "default": null,
        "comment": "字段注释"
      }}
    ],
    "relations": [
      {{
        "type": "belongs_to",
        "target_table": "关联目标表名",
        "source_cols": ["本表外键字段"],
        "target_cols": ["目标表字段"]
      }}
    ]
  }}
]

要求：
1. 未明确类型时根据字段名与常识推断 MySQL 类型。
2. 若提到表间关联，必须在 relations 中写出。
3. 仅返回 JSON 数组，不要 Markdown 或解释文字。

用户描述：
{user_text}
"""


class SchemaManager:
    """管理 schema_docs（Milvus）与 Table、Column、RELATES_TO 图（Neo4j）。"""

    def __init__(self, memory_system: MemorySystem) -> None:
        self.memory = memory_system
        self.schema_col: Optional[Collection] = None
        self._init_schema_collection()
        self._ensure_schema_constraints()

    def _assert_schema_embedding_dim(self, col: Collection) -> None:
        actual = _milvus_collection_embedding_dim(col)
        if actual is None:
            return
        expected = int(settings.vector_dim)
        if actual != expected:
            raise RuntimeError(
                f'Milvus 集合「{SCHEMA_COLLECTION_NAME}」向量维度为 {actual}，与 vector_dim={expected} 不一致；'
                f"请删除该集合并重启，或调整 VECTOR_DIM / 嵌入模型。"
            )

    def _init_schema_collection(self) -> None:
        if utility.has_collection(SCHEMA_COLLECTION_NAME):
            self.schema_col = Collection(SCHEMA_COLLECTION_NAME)
            self.schema_col.load()
            self._assert_schema_embedding_dim(self.schema_col)
            return

        dim = int(settings.vector_dim)
        fields = [
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="table_name", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="confidence", dtype=DataType.FLOAT),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="updated_at", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, description="Table schema documents with metadata")
        self.schema_col = Collection(SCHEMA_COLLECTION_NAME, schema)
        self.schema_col.create_index(
            "embedding",
            {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}},
        )
        for scalar in ("table_name", "confidence"):
            try:
                self.schema_col.create_index(scalar, {"index_type": "AUTOINDEX"})
            except Exception as e:
                log.debug("创建标量索引 %s: %s", scalar, e)
        self.schema_col.load()

    def _ensure_schema_constraints(self) -> None:
        """Neo4j 社区版不支持 (table_name,name) 的 NODE KEY，改用 column_key 单列 UNIQUE。"""
        with self.memory._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT table_name_unique IF NOT EXISTS FOR (t:Table) REQUIRE t.name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT column_key_unique IF NOT EXISTS "
                "FOR (c:Column) REQUIRE c.column_key IS UNIQUE"
            )

    def parse_markdown(self, text: str) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        # 支持文件以 ## 开头（不必前置换行）
        blocks = re.split(r"(?:^|\n)##\s+", text)
        for block in blocks:
            if not block.strip():
                continue
            lines = block.strip().split("\n")
            header = re.sub(r"^#+\s*", "", lines[0].strip())
            raw_tname = header.split()[0] if header else ""
            table_name = normalize_schema_identifier(raw_tname)
            comment_match = re.search(r"[\（\(](.*?)[\）\)]", header)
            table_comment = (comment_match.group(1) or "").strip() if comment_match else ""

            table_lines = [ln for ln in lines if ln.strip().startswith("|")]
            if len(table_lines) < 1:
                continue

            row1_parts = [p.strip() for p in table_lines[1].split("|")[1:-1]] if len(table_lines) > 1 else []
            if len(table_lines) >= 2 and _is_md_separator_row(row1_parts):
                header_cells = [p.strip() for p in table_lines[0].split("|")[1:-1]]
                idx = _markdown_header_indices(header_cells)
                if "name" not in idx:
                    idx = {"name": 0, "type": 1, "comment": 2}
                data_rows = table_lines[2:]
            else:
                idx = {"name": 0, "type": 1, "comment": 2, "key": 3, "link": 4}
                data_rows = []
                for ln in table_lines:
                    pp = [p.strip() for p in ln.split("|")[1:-1]]
                    if _is_md_separator_row(pp):
                        continue
                    data_rows.append(ln)

            columns: List[Dict[str, Any]] = []
            relations: List[Dict[str, Any]] = []
            for row in data_rows:
                parts = [p.strip() for p in row.split("|")[1:-1]]
                if len(parts) < 3:
                    continue
                ni, ti, ci = idx.get("name", 0), idx.get("type", 1), idx.get("comment", 2)
                if ni >= len(parts) or ti >= len(parts) or ci >= len(parts):
                    continue
                col_name, col_type, col_comment = parts[ni], parts[ti], parts[ci]
                key = ""
                if "key" in idx and idx["key"] < len(parts):
                    key = parts[idx["key"]].strip().upper()
                if key not in ("PRI", "UNI", "MUL", ""):
                    key = ""
                link_cell = ""
                if "link" in idx and idx["link"] < len(parts):
                    link_cell = parts[idx["link"]]
                columns.append(
                    {
                        "name": col_name,
                        "type": col_type,
                        "nullable": True,
                        "key": key,
                        "default": None,
                        "comment": col_comment,
                    }
                )
                rel = _relation_from_link_cell(col_name, link_cell)
                if rel:
                    relations.append(rel)

            if table_name and columns:
                tables.append(
                    {
                        "name": table_name,
                        "comment": table_comment,
                        "columns": columns,
                        "relations": _dedupe_relations(relations),
                    }
                )
        return _finalize_tables_list(tables)

    def _excel_resolve_columns(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """将 DataFrame 列名映射到标准键；缺必须列返回 None。"""
        norm: Dict[str, str] = {}
        for c in df.columns:
            norm[str(c).strip().lower().replace(" ", "")] = str(c)

        def pick(*keys: str) -> Optional[str]:
            for k in keys:
                if k in norm:
                    return norm[k]
            return None

        c_name = pick("字段名", "field", "column", "列名")
        c_type = pick("类型", "type")
        c_comment = pick("注释", "comment", "说明", "描述", "description")
        if not c_name or not c_type or not c_comment:
            return None
        return {
            "name": c_name,
            "type": c_type,
            "comment": c_comment,
            "nullable": pick("是否为空", "nullable"),
            "key": pick("键", "key"),
            "default": pick("默认值", "default"),
            "relation": pick("关联关系", "relation", "foreignkey", "外键", "关联"),
        }

    def parse_excel(self, file_content: bytes) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []
        xl = pd.ExcelFile(BytesIO(file_content))
        for sheet_name in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet_name, header=0)
            cmap = self._excel_resolve_columns(df)
            if not cmap:
                log.warning("跳过 Sheet %r：缺少必须列（字段名/类型/注释或其英文字段）", sheet_name)
                continue
            columns: List[Dict[str, Any]] = []
            relations: List[Dict[str, Any]] = []
            c_rel = cmap.get("relation")

            def _row_scalar(row: Any, col_name: Optional[str]) -> Any:
                if not col_name:
                    return None
                v = row.get(col_name)
                if isinstance(v, pd.Series):
                    if len(v) == 0:
                        return None
                    v = v.iloc[0]
                return v

            for _, row in df.iterrows():
                nullable_val = True
                if cmap.get("nullable"):
                    nc = _row_scalar(row, cmap["nullable"])
                    if nc is not None and pd.notna(nc):
                        nullable_val = bool(nc)
                def_cell = _row_scalar(row, cmap["default"]) if cmap.get("default") else None
                key = ""
                if cmap.get("key"):
                    kv = _row_scalar(row, cmap["key"])
                    if kv is not None and pd.notna(kv):
                        key = str(kv).strip().upper()
                if key not in ("PRI", "UNI", "MUL", ""):
                    key = ""
                col_name = str(_row_scalar(row, cmap["name"]) or "").strip()
                col_type = str(_row_scalar(row, cmap["type"]) or "").strip()
                col_comment = str(_row_scalar(row, cmap["comment"]) or "").strip()
                if not col_name:
                    continue
                columns.append(
                    {
                        "name": col_name,
                        "type": col_type,
                        "nullable": nullable_val,
                        "key": key,
                        "default": def_cell if def_cell is not None and pd.notna(def_cell) else None,
                        "comment": col_comment,
                    }
                )
                if c_rel:
                    rv = _row_scalar(row, c_rel)
                    if rv is not None and not (isinstance(rv, float) and pd.isna(rv)):
                        rel = _relation_from_link_cell(col_name, str(rv))
                        if rel:
                            relations.append(rel)
            tname = normalize_schema_identifier(str(sheet_name))
            if tname and columns:
                tables.append(
                    {
                        "name": tname,
                        "comment": "",
                        "columns": columns,
                        "relations": _dedupe_relations(relations),
                    }
                )
        return _finalize_tables_list(tables)

    def parse_natural_language_tables(self, user_text: str) -> List[Dict[str, Any]]:
        """LLM 从自然语言提取表结构 JSON 数组；失败返回 []."""
        ut = (user_text or "").strip()
        if not ut:
            return []
        prompt = NL_TABLES_PROMPT.format(user_text=ut)
        raw = self._call_llm_for_sql_raw(prompt)
        for attempt in range(2):
            cleaned = _strip_json_fence(raw)
            try:
                data = json.loads(cleaned)
                if not isinstance(data, list):
                    return []
                return _finalize_tables_list(data)
            except json.JSONDecodeError as e:
                log.warning("自然语言表结构 JSON 解析失败 attempt=%s: %s", attempt + 1, e)
                if attempt == 0:
                    raw = self._call_llm_for_sql_raw(
                        prompt
                        + "\n\n上次输出无法被 json.loads 解析。请仅输出合法 JSON 数组，不要其它文字。"
                    )
                else:
                    break
        return []

    def ingest_tables(
        self,
        tables: List[Dict[str, Any]],
        source_file: str,
        user_text: str = "",
        confidence: float = 0.6,
        source_type: str = "upload",
    ) -> Dict[str, Any]:
        tables = _finalize_tables_list([t for t in tables if isinstance(t, dict)])
        if not tables:
            return {"success": False, "message": "未解析到任何表结构"}

        timestamp = int(time.time())
        milvus_rows: List[Dict[str, Any]] = []

        with self.memory._driver.session() as session:
            for table in tables:
                lines = [f"表名：{table['name']}"]
                if table.get("comment"):
                    lines.append(f"注释：{table['comment']}")
                lines.append("字段：")
                for col in table["columns"]:
                    line = f"  - {col['name']} ({col['type']})"
                    if col.get("comment"):
                        line += f" : {col['comment']}"
                    if col.get("key"):
                        line += f" [{col['key']}]"
                    lines.append(line)
                rels = table.get("relations") or []
                if rels:
                    lines.append("关联关系：")
                    for rel in rels:
                        sc = ",".join(rel.get("source_cols") or [])
                        tc = ",".join(rel.get("target_cols") or [])
                        lines.append(
                            f"  - 通过 [{sc}] 关联 {rel['target_table']}({tc}) [类型：{rel.get('type', 'belongs_to')}]"
                        )
                if user_text:
                    lines.append(f"补充说明：{user_text}")
                doc_text = "\n".join(lines)

                vec = self.memory._get_embedding(doc_text)
                doc_id = f"doc_{self.memory._id_gen.next_str()}"

                milvus_rows.append(
                    {
                        "doc_id": doc_id,
                        "content": doc_text[:4096],
                        "embedding": vec,
                        "source_file": source_file[:256],
                        "table_name": str(table["name"])[:64],
                        "timestamp": timestamp,
                        "confidence": float(confidence),
                        "source": (source_type or "upload")[:256],
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    }
                )

                session.run(
                    """
                    MERGE (t:Table {name: $name})
                    SET t.comment = $comment,
                        t.source_doc = $source,
                        t.confidence = $confidence,
                        t.source_type = $source_type,
                        t.updated_at = $ts
                    """,
                    name=table["name"],
                    comment=table.get("comment", ""),
                    source=source_file,
                    confidence=float(confidence),
                    source_type=source_type,
                    ts=timestamp,
                )
                for col in table["columns"]:
                    col_name = col["name"]
                    column_key = f"{table['name']}|{col_name}"
                    session.run(
                        """
                        MATCH (t:Table {name: $table})
                        MERGE (c:Column {column_key: $column_key})
                        SET c.table_name = $table, c.name = $col_name, c.type = $type,
                            c.nullable = $nullable, c.key = $key,
                            c.default = $default, c.comment = $comment
                        MERGE (t)-[:HAS_COLUMN]->(c)
                        """,
                        table=table["name"],
                        column_key=column_key,
                        col_name=col_name,
                        type=col["type"],
                        nullable=col.get("nullable", True),
                        key=col.get("key", ""),
                        default=col.get("default"),
                        comment=col.get("comment", ""),
                    )

                for rel in table.get("relations") or []:
                    from_t = table["name"]
                    to_t = rel["target_table"]
                    sc = list(rel.get("source_cols") or [])
                    tc = list(rel.get("target_cols") or [])
                    if not sc or not tc:
                        continue
                    rk = f"{from_t}|{to_t}|{','.join(sc)}|{','.join(tc)}|{rel.get('type', 'belongs_to')}"
                    session.run(
                        """
                        MERGE (a:Table {name: $from_table})
                        MERGE (b:Table {name: $to_table})
                        MERGE (a)-[r:RELATES_TO {rel_key: $rk}]->(b)
                        SET r.type = $rtype,
                            r.source_cols = $sc,
                            r.target_cols = $tc,
                            r.left_col = $lc,
                            r.right_col = $rc,
                            r.updated_at = $ts
                        """,
                        from_table=from_t,
                        to_table=to_t,
                        rk=rk,
                        rtype=str(rel.get("type", "belongs_to")),
                        sc=sc,
                        tc=tc,
                        lc=sc[0],
                        rc=tc[0],
                        ts=timestamp,
                    )

        assert self.schema_col is not None
        field_names = [
            "doc_id",
            "content",
            "embedding",
            "source_file",
            "table_name",
            "timestamp",
            "confidence",
            "source",
            "created_at",
            "updated_at",
        ]
        data_cols: List[List[Any]] = [[row[f] for row in milvus_rows] for f in field_names]
        self.schema_col.insert(data_cols)
        self.schema_col.flush()

        return {
            "success": True,
            "tables_imported": [t["name"] for t in tables],
            "message": f"已导入 {len(tables)} 张表，置信度 {confidence}",
        }

    def search_schema_tables(
        self,
        natural_language: str,
        *,
        min_confidence: Optional[float] = None,
        limit: int = 8,
    ) -> Dict[str, Any]:
        """Milvus 语义检索 schema_docs，去重返回候选表及片段（供 ReAct search_tables）。"""
        assert self.schema_col is not None
        vec = self.memory._get_embedding(natural_language)
        expr: Optional[str] = None
        if min_confidence is not None:
            expr = f"confidence >= {float(min_confidence)}"
        hits = self.schema_col.search(
            data=[vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": int(settings.milvus_search_ef)}},
            limit=max(1, int(limit)),
            expr=expr,
            output_fields=["content", "table_name", "confidence"],
        )
        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for hit in hits[0]:
            tn = hit.entity.get("table_name")
            if not tn or str(tn) in seen:
                continue
            seen.add(str(tn))
            content = hit.entity.get("content") or ""
            rows.append(
                {
                    "table_name": str(tn),
                    "content_excerpt": str(content)[:800],
                    "confidence": hit.entity.get("confidence"),
                }
            )
        return {"tables": rows, "empty": len(rows) == 0}

    def load_tables_info(self, table_names: List[str]) -> List[Dict[str, Any]]:
        """从 Neo4j 批量加载多张表的字段元数据。"""
        tables_info: List[Dict[str, Any]] = []
        with self.memory._driver.session() as session:
            for tname in table_names:
                rec = session.run(
                    """
                    MATCH (t:Table {name: $name})
                    OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
                    RETURN t.name AS name, t.comment AS comment,
                           collect(DISTINCT {
                             name: c.name, type: c.type, comment: c.comment, key: c.key
                           }) AS columns
                    """,
                    name=tname,
                ).single()
                if rec and rec.get("name"):
                    cols = [c for c in (rec["columns"] or []) if c and c.get("name")]
                    tables_info.append(
                        {
                            "name": rec["name"],
                            "comment": rec["comment"],
                            "columns": cols,
                        }
                    )
        return tables_info

    def describe_table_graph(self, table_name: str) -> Dict[str, Any]:
        """单表结构与注释（供 ReAct describe_table）。"""
        info = self.load_tables_info([table_name])
        if not info:
            return {"error": f"未知表或未导入图谱: {table_name!r}"}
        return {"table": info[0]}

    def get_foreign_keys_graph(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """从 Neo4j 读取 RELATES_TO（若未建模则返回空列表）。"""
        with self.memory._driver.session() as session:
            if table_name:
                recs = session.run(
                    """
                    MATCH (a:Table)-[r:RELATES_TO]->(b:Table)
                    WHERE a.name = $t OR b.name = $t
                    RETURN a.name AS from_table, b.name AS to_table,
                           coalesce(r.type, '') AS rel_type,
                           coalesce(r.left_col, '') AS left_col,
                           coalesce(r.right_col, '') AS right_col,
                           coalesce(r.source_cols, []) AS source_cols,
                           coalesce(r.target_cols, []) AS target_cols
                    """,
                    t=table_name,
                )
            else:
                recs = session.run(
                    """
                    MATCH (a:Table)-[r:RELATES_TO]->(b:Table)
                    RETURN a.name AS from_table, b.name AS to_table,
                           coalesce(r.type, '') AS rel_type,
                           coalesce(r.left_col, '') AS left_col,
                           coalesce(r.right_col, '') AS right_col,
                           coalesce(r.source_cols, []) AS source_cols,
                           coalesce(r.target_cols, []) AS target_cols
                    """
                )
            rels = [dict(r) for r in recs]
        return {"relationships": rels, "note": "若无数据表示尚未导入表间关系到 Neo4j。"}

    def execute_readonly_sql(self, sql: str, mode: str) -> Dict[str, Any]:
        """只读 SQL：dry_run 仅校验不连库执行；auto_execute 连 MySQL 执行。"""
        s = (sql or "").strip()
        if not s:
            return {"error": "SQL 为空"}
        if not is_safe_sql(s):
            return {"error": "SQL 未通过安全校验（非只读或多语句等）", "sql": s}
        if mode == "dry_run":
            return {
                "skipped": True,
                "message": "dry_run：已校验 SQL，未对 MySQL 执行；若需真实结果请使用 mode=auto_execute。",
                "sql": s,
            }
        if mode != "auto_execute":
            return {"error": f"未知 mode: {mode!r}，仅支持 dry_run / auto_execute", "sql": s}
        try:
            engine = create_engine(settings.mysql_url)
            with engine.connect() as conn:
                rows = conn.execute(text(s)).mappings().all()
                data = [dict(r) for r in rows]
                if len(data) > 500:
                    return {
                        "truncated": True,
                        "row_count": len(data),
                        "rows": data[:500],
                        "message": "结果超过 500 行已截断",
                    }
                return {"rows": data, "row_count": len(data)}
        except Exception as e:
            return {"error": f"SQL 执行失败: {e}", "sql": s}

    def query_schema(
        self,
        user_query: str,
        mode: str = "dry_run",
        min_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        assert self.schema_col is not None
        search = self.search_schema_tables(user_query, min_confidence=min_confidence, limit=5)
        if search.get("empty"):
            return {"error": "未找到相关表，请检查知识库是否已导入"}

        relevant_tables = [r["table_name"] for r in search["tables"]]
        tables_info = self.load_tables_info(relevant_tables)
        if not tables_info:
            return {"error": "未找到相关表，请检查知识库是否已导入"}

        prompt = self._build_sql_prompt(user_query, tables_info)
        raw_llm = self._call_llm_for_sql_raw(prompt)
        sql, explanation = parse_sql_explanation_from_llm(raw_llm)
        if not sql:
            sql = raw_llm.strip()
            sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
            sql = re.sub(r"\s*```$", "", sql).strip()
            explanation = explanation or ""

        if not is_safe_sql(sql):
            return {"error": "生成的 SQL 包含危险操作或多语句，已拦截", "sql": sql, "explanation": explanation}

        result: Dict[str, Any] = {"sql": sql, "explanation": explanation or "（模型未返回说明）"}
        exec_out = self.execute_readonly_sql(sql, mode)
        if exec_out.get("error"):
            result["error"] = exec_out["error"]
        elif exec_out.get("skipped"):
            pass
        elif "rows" in exec_out:
            result["result"] = exec_out["rows"]
        elif exec_out.get("truncated"):
            result["result"] = exec_out.get("rows", [])
            result["warning"] = exec_out.get("message")

        return result

    def _build_sql_prompt(self, question: str, tables: List[Dict[str, Any]]) -> str:
        schema_desc = ""
        for t in tables:
            schema_desc += f"\n表 {t['name']}"
            if t.get("comment"):
                schema_desc += f"（{t['comment']}）"
            schema_desc += "\n字段：\n"
            for col in t["columns"]:
                line = f"  {col['name']} {col['type']}"
                if col.get("comment"):
                    line += f"  -- {col['comment']}"
                if col.get("key") == "PRI":
                    line += " (主键)"
                schema_desc += line + "\n"

        return f"""你是一个 MySQL 专家。根据以下表结构信息，将用户问题转换为一条只读的 MySQL 语句
（仅 SELECT / WITH … SELECT / SHOW / DESCRIBE / DESC / EXPLAIN）。

请严格按下面两行输出（不要输出 Markdown 代码块以外的多余段落）：
SQL: <一条语句，不要写多条，不要分号拼接危险语句>
EXPLANATION: <一句中文说明该语句在回答什么问题>

表结构：
{schema_desc}

用户问题：{question}
"""

    def _call_llm_for_sql_raw(self, prompt: str) -> str:
        client = ollama.Client(
            host=settings.ollama_base_url,
            timeout=settings.ollama_request_timeout_s,
        )
        resp = client.chat(
            model=settings.ollama_chat_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.get("message") or {}).get("content") or ""
