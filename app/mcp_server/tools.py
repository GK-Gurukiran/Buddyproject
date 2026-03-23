"""
MCP (Model Context Protocol) tool definitions.
Exposes web search and scraping capabilities via the MCP standard.

Primary: Tavily / Firecrawl when API keys configured.
Fallback: DuckDuckGo / httpx+BeautifulSoup (free, no keys).

Note: The MCP server requires Python 3.10+. These tool handlers
are also used directly by the agent system regardless of MCP availability.
"""

import logging
from typing import Dict, List, Any
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Try to import MCP types, fall back to plain dicts if unavailable
try:
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Tool = dict
    TextContent = dict


# ──────────────────────────────────────────────
# MCP Tool Definitions
# ──────────────────────────────────────────────

if MCP_AVAILABLE:
    MCP_TOOLS = [
        Tool(
            name="web_search",
            description="Search the web for current information. Uses Tavily or DuckDuckGo.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "max_results": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="scrape_url",
            description="Scrape and extract content from a webpage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to scrape"},
                },
                "required": ["url"],
            },
        ),
        Tool(
            name="search_and_scrape",
            description="Search the web then scrape the top result.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        ),
    ]
else:
    MCP_TOOLS = [
        {"name": "web_search", "description": "Search the web (Tavily or DuckDuckGo)."},
        {"name": "scrape_url", "description": "Scrape a webpage (Firecrawl or httpx)."},
        {"name": "search_and_scrape", "description": "Search then scrape top result."},
    ]


# ──────────────────────────────────────────────
# Free fallback helpers
# ──────────────────────────────────────────────

def _duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Free web search using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        formatted = []
        for r in results:
            formatted.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('href', r.get('link', ''))}\n"
                f"Content: {r.get('body', r.get('snippet', 'No content'))}\n"
            )
        return "\n---\n".join(formatted)
    except ImportError:
        return "DuckDuckGo not available. Install: pip install duckduckgo-search"
    except Exception as e:
        return f"DuckDuckGo error: {str(e)}"


def _httpx_scrape(url: str) -> str:
    """Free web scraper using httpx + BeautifulSoup."""
    try:
        import httpx
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ScalableChatbot/1.0)"}
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find("body") or soup
        text = main.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        content = "\n".join(lines)
        if len(content) > 8000:
            content = content[:8000] + "\n\n... [truncated]"
        return content
    except ImportError:
        return "Scraping not available. Install: pip install httpx beautifulsoup4"
    except Exception as e:
        return f"Scrape error: {str(e)}"


# ──────────────────────────────────────────────
# Tool Execution Handlers
# ──────────────────────────────────────────────

async def execute_web_search(query: str, max_results: int = 5) -> str:
    """Execute web search — Tavily first, then DuckDuckGo fallback."""
    if settings.tavily_api_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.tavily_api_key)
            response = client.search(query=query, max_results=max_results, search_depth="advanced")
            results = []
            for r in response.get("results", []):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('url', '')}\n"
                    f"Content: {r.get('content', 'No content')}\n"
                )
            return "\n---\n".join(results) if results else "No results found."
        except Exception as e:
            logger.warning(f"Tavily failed ({e}), using DuckDuckGo fallback")

    return _duckduckgo_search(query, max_results)


async def execute_scrape_url(url: str) -> str:
    """Execute web scraping — Firecrawl first, then httpx fallback."""
    if settings.firecrawl_api_key:
        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=settings.firecrawl_api_key)
            result = app.scrape_url(url, params={"formats": ["markdown"]})
            content = result.get("markdown", result.get("content", "No content extracted."))
            if len(content) > 8000:
                content = content[:8000] + "\n\n... [truncated]"
            return content
        except Exception as e:
            logger.warning(f"Firecrawl failed ({e}), using httpx fallback")

    return _httpx_scrape(url)


async def execute_search_and_scrape(query: str) -> str:
    """Search the web, then scrape the top result."""
    search_result = await execute_web_search(query, max_results=1)

    url = None
    for line in search_result.split("\n"):
        if line.startswith("URL:"):
            url = line.replace("URL:", "").strip()
            break

    if not url:
        return f"Search results:\n{search_result}\n\n(No URL found to scrape)"

    scrape_result = await execute_scrape_url(url)
    return f"Search results:\n{search_result}\n\nScraped content from {url}:\n{scrape_result}"


# Handler dispatch map
TOOL_HANDLERS = {
    "web_search": lambda args: execute_web_search(args["query"], args.get("max_results", 5)),
    "scrape_url": lambda args: execute_scrape_url(args["url"]),
    "search_and_scrape": lambda args: execute_search_and_scrape(args["query"]),
}
