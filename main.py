import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from config import settings
from memory_agent import run_query_assistant, run_store_assistant
from memory_service import MemorySystem

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
    app.state.memory_startup_error: Optional[str] = None
    try:
        ms.connect()
        app.state.memory = ms
        log.info("启动：记忆后端已就绪")
    except Exception as e:
        app.state.memory_startup_error = str(e)
        log.exception("启动：记忆后端连接失败: %s", e)
    yield
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
