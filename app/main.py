import os
import json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.schemas import AskRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAVICON_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
  <rect width='64' height='64' rx='12' fill='#111827'/>
  <path d='M18 42h28' stroke='#FBBF24' stroke-width='6' stroke-linecap='round'/>
  <path d='M22 28c0-8 5-12 10-12s10 4 10 12c0 8-5 12-10 12s-10-4-10-12Z' fill='#F9FAFB' stroke='#FBBF24' stroke-width='4'/>
</svg>"""


def configure_env(project_root: Path) -> None:
    """
    Configure env vars before importing transformers-related libraries.
    This makes model cache stable across runs and reduces noisy logs.
    """
    cache_root = project_root / "data" / "models"
    cache_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


configure_env(PROJECT_ROOT)

# Load .env once (DASHSCOPE_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    # dotenv is optional; if not installed, env vars must be set by the shell.
    pass


def sse_pack(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from scripts.prepare_assets import prepare_assets
    from rag.config import RagConfig
    from core.llm_client import LLMClient, LLMConfig
    from agent.flow_rag import RagFlow

    from agent.tools.registry import ToolRegistry
    from agent.tools.rag_tools import RagSearchTool, RagOpenTool, RagAnswerTool, RagQATool
    from agent.tools.mcp_adapter import MCPConfig, register_mcp_tools, register_dashscope_tools
    from agent.planner import LLMPlanner
    from agent.runtime import AgentRuntime

    prepare_assets()

    cfg = RagConfig()
    llm = LLMClient(LLMConfig())
    flow = RagFlow(cfg=cfg, llm=llm)

    # Warm up models so first request does not download anything.
    # This uses HuggingFace cache directory set in configure_env().
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(cfg.embedding_model_name)
    except Exception:
        pass

    try:
        from transformers import AutoTokenizer, AutoModel
        AutoTokenizer.from_pretrained(cfg.colbert_model_name)
        AutoModel.from_pretrained(cfg.colbert_model_name)
    except Exception:
        pass

    registry = ToolRegistry()
    registry.register(RagSearchTool(cfg=cfg))
    registry.register(RagOpenTool(project_root=PROJECT_ROOT))
    registry.register(RagAnswerTool(llm=llm))
    registry.register(RagQATool(flow=flow))

    # Dynamically register MCP-provided tools if configured.
    mcp_cfg = MCPConfig.from_env()
    register_mcp_tools(registry, mcp_cfg)
    register_dashscope_tools(registry)

    planner = LLMPlanner(llm=llm)
    agent = AgentRuntime(registry=registry, planner=planner, llm=llm)

    app.state.cfg = cfg
    app.state.llm = llm
    app.state.flow = flow
    app.state.agent = agent

    yield


app = FastAPI(lifespan=lifespan)

STATIC_DIR = PROJECT_ROOT / "app" / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    agent = app.state.agent

    def gen() -> Iterator[str]:
        for ev in agent.run_stream(
            req.message,
            session_id=req.session_id,
            debug=req.debug,
            use_colbert=req.use_colbert,
            rag_enabled=req.rag_enabled,
            agent_enabled=req.agent_enabled,
            web_enabled=req.web_enabled,
            fusion_mode=req.fusion_mode,
            max_steps=req.max_steps,
        ):
            etype = ev.get("type", "message")

            if etype == "delta":
                yield sse_pack("delta", {"text": ev.get("text", "")})
            elif etype == "reasoning":
                yield sse_pack("reasoning", {"text": ev.get("text", "")})
            elif etype == "meta":
                meta = dict(ev)
                meta.pop("type", None)
                yield sse_pack("meta", meta)
            elif etype == "usage":
                yield sse_pack("usage", {"usage": ev.get("usage", {})})
            elif etype == "done":
                done = dict(ev)
                done.pop("type", None)
                yield sse_pack("done", done)
            elif etype == "error":
                yield sse_pack("error", {"message": ev.get("message", "unknown error")})
            elif etype == "agent_meta":
                other = dict(ev)
                other.pop("type", None)
                yield sse_pack("agent_meta", other)
            elif etype == "tool_start":
                other = dict(ev)
                other.pop("type", None)
                yield sse_pack("tool_start", other)
            elif etype == "tool_end":
                other = dict(ev)
                other.pop("type", None)
                yield sse_pack("tool_end", other)
            elif etype == "plan_update":
                other = dict(ev)
                other.pop("type", None)
                yield sse_pack("plan_update", other)
            else:
                other = dict(ev)
                other.pop("type", None)
                yield sse_pack("message", other)

    return StreamingResponse(gen(), media_type="text/event-stream")
