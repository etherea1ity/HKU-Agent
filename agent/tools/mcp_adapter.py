from __future__ import annotations

import os
import json
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.tools.base import ToolContext, ToolResult
from agent.tools.registry import ToolRegistry

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

# Optional MCP SDK imports (for DashScope IQS MCP server)
try:
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
except Exception:  # pragma: no cover - optional dependency
    httpx = None
    ClientSession = None
    streamable_http_client = None


@dataclass
class MCPConfig:
    """Config for talking to an MCP server."""

    server_url: Optional[str] = None
    api_key_env: str = "MCP_API_KEY"
    timeout: float = 15.0

    @classmethod
    def from_env(cls) -> "MCPConfig":
        return cls(server_url=os.getenv("MCP_SERVER_URL"))

    @property
    def enabled(self) -> bool:
        return bool(self.server_url)


def _auth_headers(cfg: MCPConfig) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = os.getenv(cfg.api_key_env)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


@dataclass
class MCPToolSpec:
    name: str
    remote_name: str
    description: str
    input_schema: Dict[str, Any]


class MCPClient:
    """Minimal HTTP client for MCP-like tool listing and invocation."""

    def __init__(self, cfg: MCPConfig):
        if not cfg.enabled:
            raise RuntimeError("MCP server_url not configured")
        if requests is None:
            raise RuntimeError("requests is required for MCP support; pip install requests")
        self.cfg = cfg

    def _url(self, path: str) -> str:
        base = (self.cfg.server_url or "").rstrip("/")
        return f"{base}{path}"

    def list_tools(self) -> List[MCPToolSpec]:
        resp = requests.get(self._url("/tools"), headers=_auth_headers(self.cfg), timeout=self.cfg.timeout)
        resp.raise_for_status()
        data = resp.json() or []
        specs: List[MCPToolSpec] = []
        for item in data:
            remote_name = item.get("name")
            if not remote_name:
                continue
            specs.append(
                MCPToolSpec(
                    name=f"mcp.{remote_name}",
                    remote_name=remote_name,
                    description=item.get("description") or "MCP tool",
                    input_schema=item.get("input_schema") or item.get("schema") or {},
                )
            )
        return specs

    def call_tool(self, remote_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(
            self._url("/call"),
            headers=_auth_headers(self.cfg),
            json={"name": remote_name, "args": args},
            timeout=self.cfg.timeout,
        )
        resp.raise_for_status()
        return resp.json() or {}


class MCPAdapter:
    """Discovers MCP tools and exposes them as local Tool implementations."""

    def __init__(self, cfg: MCPConfig):
        self.cfg = cfg
        self.client = MCPClient(cfg)

    def discover(self) -> List["MCPProxyTool"]:
        specs = self.client.list_tools()
        return [MCPProxyTool(self, spec) for spec in specs]

    def call(self, spec: MCPToolSpec, args: Dict[str, Any]) -> ToolResult:
        try:
            result = self.client.call_tool(spec.remote_name, args)
            return ToolResult(
                ok=bool(result.get("ok", True)),
                content=result.get("content") or result.get("text"),
                artifacts=result.get("artifacts"),
                debug=result.get("debug"),
                error=result.get("error"),
            )
        except Exception as e:  # keep failures contained to the tool
            return ToolResult(ok=False, error=str(e))


class MCPProxyTool:
    """A single MCP tool exposed through the local Tool protocol."""

    def __init__(self, adapter: MCPAdapter, spec: MCPToolSpec):
        self.adapter = adapter
        self.spec = spec
        self.name = spec.name
        self.description = spec.description
        self.input_schema = spec.input_schema

    def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        # Pass through user-provided args to MCP server.
        return self.adapter.call(self.spec, kwargs)


def register_mcp_tools(registry: ToolRegistry, cfg: MCPConfig) -> List[str]:
    """Discover MCP tools (if configured) and register them. Returns registered names."""
    if not cfg.enabled:
        return []
    try:
        adapter = MCPAdapter(cfg)
        tools = adapter.discover()
    except Exception:
        return []

    registered: List[str] = []
    for t in tools:
        try:
            registry.register(t)
            registered.append(t.name)
        except Exception:
            # Skip duplicates or bad schemas without failing startup.
            continue
    return registered


# -------------------- DashScope IQS MCP (Aliyun) via MCP SDK --------------------


SEARCH_MCP_URL = "https://iqs-mcp.aliyuncs.com/mcp-servers/iqs-mcp-server-search"
READPAGE_MCP_URL = "https://iqs-mcp.aliyuncs.com/mcp-servers/iqs-mcp-server-readpage"


def _dashscope_key() -> Optional[str]:
    return os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY") or os.getenv("MCP_API_KEY")


def _need_mcp_sdk() -> Optional[str]:
    if httpx is None or ClientSession is None or streamable_http_client is None:
        return "Missing mcp/httpx packages; install with: pip install mcp httpx"
    return None


async def _call_iqs_tool(url: str, tool_name: str, api_key: str, arguments: Dict[str, Any]) -> ToolResult:
    err = _need_mcp_sdk()
    if err:
        return ToolResult(ok=False, error=err)

    headers = {"X-API-Key": api_key, "Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(headers=headers, timeout=30.0) as http_client:
            async with streamable_http_client(url, http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    tool_names = [t.name for t in tools.tools]
                    if tool_name not in tool_names:
                        return ToolResult(ok=False, error=f"Tool {tool_name} not found. Available: {tool_names}")

                    result = await session.call_tool(tool_name, arguments=arguments)

                    # Extract text content blocks
                    texts: List[str] = []
                    try:
                        for block in getattr(result, "content", []) or []:
                            if getattr(block, "type", None) == "text":
                                txt = getattr(block, "text", None)
                                if txt:
                                    texts.append(str(txt))
                    except Exception:
                        pass

                    structured = getattr(result, "structuredContent", None)
                    if structured:
                        try:
                            texts.append(json.dumps(structured, ensure_ascii=False))
                        except Exception:
                            pass

                    artifacts = {
                        "tool_names": tool_names,
                        "structured": structured,
                    }

                    content = "\n".join(texts).strip()
                    return ToolResult(ok=True, content=content, artifacts=artifacts)
    except Exception as e:
        return ToolResult(ok=False, error=str(e))


def _call_iqs_tool_sync(url: str, tool_name: str, api_key: str, arguments: Dict[str, Any]) -> ToolResult:
    return asyncio.run(_call_iqs_tool(url, tool_name, api_key, arguments))


def register_dashscope_tools(registry: ToolRegistry) -> List[str]:
    """Register DashScope IQS WebSearch/WebParser using MCP SDK (if API key present)."""
    api_key = _dashscope_key()
    if not api_key:
        return []

    specs = [
        {
            "name": "mcp.web_search",
            "mcp_url": SEARCH_MCP_URL,
            "remote_tool": "common_search",
            "desc": "DashScope IQS WebSearch (internet search)",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "recency": {"type": "string"},
                    "domains": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query"],
            },
        },
        {
            "name": "mcp.web_parser",
            "mcp_url": READPAGE_MCP_URL,
            "remote_tool": "readpage_basic",
            "desc": "DashScope IQS WebParser (fetch & parse webpage)",
            "schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    ]

    registered: List[str] = []
    err = _need_mcp_sdk()
    if err:
        return registered

    for spec in specs:
        class _DashscopeTool:
            def __init__(self, name: str, mcp_url: str, remote_tool: str, desc: str, schema: Dict[str, Any]):
                self.name = name
                self.mcp_url = mcp_url
                self.remote_tool = remote_tool
                self.description = desc
                self.input_schema = schema

            def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
                return _call_iqs_tool_sync(self.mcp_url, self.remote_tool, api_key, kwargs)

        try:
            tool = _DashscopeTool(spec["name"], spec["mcp_url"], spec["remote_tool"], spec["desc"], spec["schema"])
            registry.register(tool)
            registered.append(tool.name)
        except Exception:
            continue

    return registered
