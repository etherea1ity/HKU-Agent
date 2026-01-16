from __future__ import annotations

import asyncio
import json
import os

import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client


SEARCH_SERVER_URL = "https://iqs-mcp.aliyuncs.com/mcp-servers/iqs-mcp-server-search"
READPAGE_SERVER_URL = "https://iqs-mcp.aliyuncs.com/mcp-servers/iqs-mcp-server-readpage"


def _print_call_result(result: types.CallToolResult, title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    # structuredContent 优先（如果有）
    structured = getattr(result, "structuredContent", None)
    if structured:
        print("[structuredContent]")
        print(json.dumps(structured, ensure_ascii=False, indent=2))

    # 再打印 content（常见是 TextContent）
    if getattr(result, "content", None):
        print("\n[content]")
        for i, c in enumerate(result.content):
            if isinstance(c, types.TextContent):
                print(f"- [{i}] TextContent:\n{c.text[:4000]}")
            else:
                # 其他类型（图片/嵌入资源等）
                print(f"- [{i}] {type(c)}: {c}")


async def call_search(query: str) -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    print(api_key)
    if not api_key:
        raise RuntimeError("Missing env var: DASHSCOPE_API_KEY")

    # 阿里云文档要求：X-API-Key 头 :contentReference[oaicite:8]{index=8}
    headers = {"X-API-Key": api_key}

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        async with streamable_http_client(SEARCH_SERVER_URL, http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                print("Search server tools:", [t.name for t in tools.tools])

                # 联网搜索工具：common_search，必填 query :contentReference[oaicite:9]{index=9}
                result = await session.call_tool(
                    "common_search",
                    arguments={
                        "query": query,
                        "count": 5,
                        "page": 1,
                        # 可选：time_range / industry ...
                    },
                )
                _print_call_result(result, f"common_search(query={query!r})")


async def call_readpage(url: str) -> None:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing env var: DASHSCOPE_API_KEY")

    headers = {"X-API-Key": api_key}

    async with httpx.AsyncClient(headers=headers, timeout=60.0) as client:
        async with streamable_http_client(READPAGE_SERVER_URL, http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                print("Readpage server tools:", [t.name for t in tools.tools])

                # 网页解析工具：readpage_basic / readpage_scrape，必填 url :contentReference[oaicite:10]{index=10}
                result = await session.call_tool(
                    "readpage_basic",
                    arguments={
                        "url": url,
                        # "options": {...}  # 可选
                    },
                )
                _print_call_result(result, f"readpage_basic(url={url!r})")


async def main() -> None:
    # 1) 测联网搜索
    await call_search("阿里云 MCP iqs-mcp-server-search common_search 参数")

    # 2) 测网页解析（你也可以换成任意网页）
    await call_readpage("https://help.aliyun.com/zh/model-studio/custom-mcp")


if __name__ == "__main__":
    asyncio.run(main())
