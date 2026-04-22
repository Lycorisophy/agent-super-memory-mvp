import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, field_validator

from config import settings
from memory_agent import run_query_assistant, run_store_assistant
from memory_service import MemorySystem
from schema_manager import SchemaManager
from schema_react_agent import run_schema_react

if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

log = logging.getLogger(__name__)


class NaturalUserInput(BaseModel):
    """仅用户自然语言；用户维度为配置 default_user_id。"""

    input: str = Field(..., min_length=1, description="用户输入")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "启动：连接 Milvus(%s:%s) Neo4j(%s) default_user_id=%s",
        settings.milvus_host,
        settings.milvus_port,
        settings.neo4j_uri,
        settings.default_user_id,
    )
    ms = MemorySystem()
    app.state.memory = None
    app.state.schema_manager = None
    app.state.memory_startup_error: Optional[str] = None
    try:
        ms.connect()
        app.state.memory = ms
        app.state.schema_manager = SchemaManager(ms)
        log.info("启动：记忆后端与 Schema 助手已就绪")
    except Exception as e:
        app.state.memory_startup_error = str(e)
        log.exception("启动：记忆后端连接失败: %s", e)
    yield
    app.state.schema_manager = None
    if app.state.memory is not None:
        log.info("关闭：断开记忆后端")
        app.state.memory.close()
        app.state.memory = None


app = FastAPI(
    title="记忆对话",
    lifespan=lifespan,
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
)


def _schema_or_503(request: Request) -> SchemaManager:
    sm: Optional[SchemaManager] = getattr(request.app.state, "schema_manager", None)
    if sm is None:
        err = getattr(request.app.state, "memory_startup_error", None) or "未知错误"
        log.warning("请求被拒绝：Schema 后端不可用: %s", err)
        raise HTTPException(
            status_code=503,
            detail=f"Schema 后端不可用: {err}",
        )
    return sm


def _memory_or_503(request: Request) -> MemorySystem:
    mem: Optional[MemorySystem] = getattr(request.app.state, "memory", None)
    if mem is None:
        err = getattr(request.app.state, "memory_startup_error", None) or "未知错误"
        log.warning("请求被拒绝：记忆后端不可用: %s", err)
        raise HTTPException(
            status_code=503,
            detail=f"记忆后端不可用: {err}",
        )
    return mem


@app.post("/memory/conversation/store")
async def memory_conversation_store(body: NaturalUserInput, request: Request):
    preview = body.input[:200] + ("…" if len(body.input) > 200 else "")
    log.info(
        "POST /memory/conversation/store user=%s input_len=%d preview=%r",
        settings.default_user_id,
        len(body.input),
        preview,
    )
    mem = _memory_or_503(request)
    try:
        out = await asyncio.to_thread(run_store_assistant, mem, body.input)
        log.info(
            "POST /memory/conversation/store 完成 tool_called=%s reply_len=%d",
            out.get("tool_called"),
            len((out.get("reply") or "")),
        )
        return out
    except Exception as e:
        log.exception("POST /memory/conversation/store Ollama 失败: %s", e)
        raise HTTPException(status_code=502, detail=f"Ollama 编排失败: {e}") from e


@app.post("/memory/conversation/query")
async def memory_conversation_query(body: NaturalUserInput, request: Request):
    preview = body.input[:200] + ("…" if len(body.input) > 200 else "")
    log.info(
        "POST /memory/conversation/query user=%s input_len=%d preview=%r",
        settings.default_user_id,
        len(body.input),
        preview,
    )
    mem = _memory_or_503(request)
    try:
        out = await asyncio.to_thread(run_query_assistant, mem, body.input)
        log.info(
            "POST /memory/conversation/query 完成 tool_called=%s reply_len=%d",
            out.get("tool_called"),
            len((out.get("reply") or "")),
        )
        return out
    except Exception as e:
        log.exception("POST /memory/conversation/query Ollama 失败: %s", e)
        raise HTTPException(status_code=502, detail=f"Ollama 编排失败: {e}") from e


class SchemaQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = Field(default="dry_run", description="dry_run 或 auto_execute")
    min_confidence: Optional[float] = None
    agent_mode: str = Field(
        default="single_pass",
        description="single_pass=一期单步生成；react=二期多步 ReAct",
    )
    max_steps: int = Field(default=12, ge=1, le=40, description="react 模式最大推理步数")

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in ("dry_run", "auto_execute"):
            raise ValueError("mode 须为 dry_run 或 auto_execute")
        return v

    @field_validator("agent_mode")
    @classmethod
    def _validate_agent_mode(cls, v: str) -> str:
        if v not in ("single_pass", "react"):
            raise ValueError("agent_mode 须为 single_pass 或 react")
        return v


@app.post("/schema/ingest")
async def schema_ingest(
    request: Request,
    file: Optional[UploadFile] = File(None),
    user_text: Optional[str] = Form(None, max_length=5000),
    confidence: Optional[float] = Form(0.6),
    source_type: Optional[str] = Form("upload"),
):
    """上传 .md / .xlsx，或仅用自然语言（LLM 提取表结构）。"""
    sm = _schema_or_503(request)

    ut = (user_text or "").strip()
    eff_confidence = float(confidence if confidence is not None else 0.6)
    eff_source = source_type or "upload"

    if file is None:
        if len(ut) < 8:
            raise HTTPException(
                status_code=400,
                detail="请上传 .md / .xlsx，或提供至少约 8 个字的自然语言表结构描述（由 LLM 解析）。",
            )

        def _nl() -> dict[str, Any]:
            parsed = sm.parse_natural_language_tables(ut)
            if not parsed:
                return {"success": False, "message": "自然语言未能解析出表结构，请检查描述或 Ollama 服务"}
            return sm.ingest_tables(
                parsed,
                source_file="natural_language",
                user_text="",
                confidence=float(confidence if confidence is not None else 0.5),
                source_type="nl_llm",
            )

        return await asyncio.to_thread(_nl)

    filename = file.filename or "upload"
    content = await file.read()

    if filename.lower().endswith(".md"):
        text = content.decode("utf-8")
        tables = sm.parse_markdown(text)
    elif filename.lower().endswith(".xlsx"):
        tables = sm.parse_excel(content)
    else:
        raise HTTPException(status_code=400, detail="仅支持 .md 或 .xlsx 文件")

    if not tables:
        raise HTTPException(status_code=400, detail="未能从文件中解析出任何表结构，请检查格式")

    def _run() -> dict[str, Any]:
        return sm.ingest_tables(
            tables,
            source_file=filename,
            user_text=ut,
            confidence=eff_confidence,
            source_type=eff_source,
        )

    return await asyncio.to_thread(_run)


@app.post("/schema/query")
async def schema_query(request: Request, body: SchemaQueryRequest):
    sm = _schema_or_503(request)

    def _run() -> dict[str, Any]:
        log.info(
            "POST /schema/query agent_mode=%s mode=%s max_steps=%s",
            body.agent_mode,
            body.mode,
            body.max_steps,
        )
        if body.agent_mode == "react":
            return run_schema_react(
                sm,
                body.query,
                mode=body.mode,
                min_confidence=body.min_confidence,
                max_steps=body.max_steps,
            )
        return sm.query_schema(
            user_query=body.query,
            mode=body.mode,
            min_confidence=body.min_confidence,
        )

    try:
        return await asyncio.to_thread(_run)
    except Exception as e:
        log.exception("POST /schema/query 失败: %s", e)
        raise HTTPException(status_code=502, detail=f"Schema 查询失败: {e}") from e
