"""
Tool definitions for agents.
Wraps web search and web scraping as LangChain tools.

Primary: Tavily (search) and Firecrawl (scraping) when API keys are configured.
Fallback: DuckDuckGo (search) and httpx+BeautifulSoup (scraping) — free, no keys needed.
"""

import logging
from langchain_core.tools import tool
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Free fallback implementations
# ──────────────────────────────────────────────

def _duckduckgo_search(query: str, max_results: int = 5) -> str:
    """Free web search using DuckDuckGo (no API key needed)."""
    try:
        # Try new package name first (ddgs), then old (duckduckgo_search)
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))

        if not results:
            logger.warning("DuckDuckGo returned no results, trying httpx fallback")
            return _httpx_search_fallback(query)

        formatted = []
        for r in results:
            title = r.get('title', 'No Title')
            url = r.get('href', r.get('link', r.get('url', '')))
            body = r.get('body', r.get('snippet', r.get('content', 'No content available.')))
            formatted.append(
                f"**{title}**\n"
                f"URL: {url}\n"
                f"{body}\n"
            )
        return "\n---\n".join(formatted)
    except ImportError:
        logger.warning("DuckDuckGo packages not installed, trying httpx fallback search")
        return _httpx_search_fallback(query)
    except Exception as e:
        logger.warning(f"DuckDuckGo search error ({e}), trying httpx fallback")
        return _httpx_search_fallback(query)


def _httpx_search_fallback(query: str) -> str:
    """Ultimate fallback: scrape DuckDuckGo HTML results using httpx.
    Uses BeautifulSoup if available, otherwise falls back to html.parser from stdlib."""
    try:
        import httpx

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        url = f"https://html.duckduckgo.com/html/?q={query}"
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=15)
        resp.raise_for_status()

        # Try BeautifulSoup first, fall back to stdlib html.parser
        try:
            from bs4 import BeautifulSoup
            return _parse_ddg_with_bs4(resp.text)
        except ImportError:
            logger.info("bs4 not available, using stdlib html.parser for DuckDuckGo results")
            return _parse_ddg_with_stdlib(resp.text)

    except ImportError:
        return "httpx is not installed. Install: pip install httpx"
    except Exception as e:
        return f"All search methods failed. Error: {str(e)}"


def _parse_ddg_with_bs4(html: str) -> str:
    """Parse DuckDuckGo HTML results using BeautifulSoup."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    results = soup.find_all("div", class_="result") or soup.find_all("div", class_="web-result")

    if not results:
        return f"Search completed but no results parsed. Raw page length: {len(html)}"

    formatted = []
    for r in results[:5]:
        title_el = r.find("a", class_="result__a") or r.find("a")
        snippet_el = r.find("a", class_="result__snippet") or r.find("td", class_="result__snippet")

        title = title_el.get_text(strip=True) if title_el else "No title"
        link = title_el.get("href", "") if title_el else ""
        snippet = snippet_el.get_text(strip=True) if snippet_el else "No snippet"

        formatted.append(f"**{title}**\nURL: {link}\n{snippet}\n")

    return "\n---\n".join(formatted) if formatted else "No results found."


def _parse_ddg_with_stdlib(html: str) -> str:
    """Parse DuckDuckGo HTML results using Python's stdlib html.parser (no dependencies)."""
    import re
    from html.parser import HTMLParser
    from html import unescape

    class DDGParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self._in_result = False
            self._in_title = False
            self._in_snippet = False
            self._current = {}
            self._current_text = []

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            classes = attrs_dict.get("class", "")

            if tag == "div" and "result " in (classes + " "):
                self._in_result = True
                self._current = {"title": "", "url": "", "snippet": ""}
            elif self._in_result:
                if tag == "a" and "result__a" in classes:
                    self._in_title = True
                    self._current["url"] = attrs_dict.get("href", "")
                    self._current_text = []
                elif tag == "a" and "result__snippet" in classes:
                    self._in_snippet = True
                    self._current_text = []
                elif tag == "td" and "result__snippet" in classes:
                    self._in_snippet = True
                    self._current_text = []

        def handle_endtag(self, tag):
            if self._in_title and tag == "a":
                self._current["title"] = "".join(self._current_text).strip()
                self._in_title = False
            elif self._in_snippet and tag in ("a", "td"):
                self._current["snippet"] = "".join(self._current_text).strip()
                self._in_snippet = False
            elif self._in_result and tag == "div" and self._current.get("title"):
                self.results.append(self._current.copy())
                self._in_result = False
                self._current = {}

        def handle_data(self, data):
            if self._in_title or self._in_snippet:
                self._current_text.append(data)

    parser = DDGParser()
    try:
        parser.feed(html)
    except Exception:
        pass

    if not parser.results:
        # Regex fallback for simpler extraction
        titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', html)
        urls = re.findall(r'class="result__a"\s+href="([^"]+)"', html)
        snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)', html)

        for i in range(min(5, len(titles))):
            title = unescape(titles[i].strip()) if i < len(titles) else "No title"
            url = urls[i] if i < len(urls) else ""
            snippet = unescape(snippets[i].strip()) if i < len(snippets) else "No snippet"
            parser.results.append({"title": title, "url": url, "snippet": snippet})

    if not parser.results:
        return f"Search completed but no results parsed. Raw page length: {len(html)}"

    formatted = []
    for r in parser.results[:5]:
        formatted.append(f"**{r['title']}**\nURL: {r['url']}\n{r['snippet']}\n")

    return "\n---\n".join(formatted) if formatted else "No results found."


def _httpx_scrape(url: str, skip_ssl_verify: bool = False) -> str:
    """Free web scraper using httpx + BeautifulSoup (no API key needed).
    Falls back to regex-based extraction if bs4 is not installed.
    skip_ssl_verify: If True, disables SSL certificate verification."""
    try:
        import httpx
        import re

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = httpx.get(url, headers=headers, follow_redirects=True, timeout=15, verify=not skip_ssl_verify)
        response.raise_for_status()

        html_text = response.text

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_text, "html.parser")

            # Remove script/style elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Try to find main content area
            main = soup.find("main") or soup.find("article") or soup.find("body")
            if main is None:
                main = soup

            text = main.get_text(separator="\n", strip=True)
        except ImportError:
            logger.info("bs4 not available, using regex-based text extraction")
            # Remove script/style blocks
            text = re.sub(r'<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '\n', text)
            # Decode HTML entities
            from html import unescape
            text = unescape(text)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        content = "\n".join(lines)

        max_len = 8000
        if len(content) > max_len:
            content = content[:max_len] + "\n\n... [content truncated]"

        return content
    except ImportError:
        return "Web scraping not available. Install: pip install httpx"
    except Exception as e:
        return f"Scraping error: {str(e)}"


# ──────────────────────────────────────────────
# LangChain tool wrappers (auto-select best backend)
# ──────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    Uses Tavily if API key is configured, otherwise falls back to DuckDuckGo (free).

    Args:
        query: The search query string.

    Returns:
        A formatted string of search results with titles, URLs, and snippets.
    """
    logger.info(f"[web_search] Searching for: {query}")

    # Try Tavily first if configured (skip placeholder keys)
    if settings.tavily_api_key and not settings.tavily_api_key.startswith("tvly-your"):
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.tavily_api_key)
            response = client.search(query=query, max_results=5, search_depth="advanced")

            results = []
            for r in response.get("results", []):
                results.append(
                    f"**{r.get('title', 'No Title')}**\n"
                    f"URL: {r.get('url', '')}\n"
                    f"{r.get('content', 'No content available.')}\n"
                )

            return "\n---\n".join(results) if results else "No search results found."
        except Exception as e:
            logger.warning(f"Tavily search failed ({e}), falling back to DuckDuckGo")

    # Fallback to DuckDuckGo (free)
    result = _duckduckgo_search(query)
    logger.info(f"[web_search] Got result length: {len(result)}")
    return result


@tool
def scrape_webpage(url: str, skip_ssl_verify: bool = False) -> str:
    """
    Scrape and extract content from a webpage.
    Uses Firecrawl if API key is configured, otherwise falls back to httpx+BeautifulSoup (free).

    Args:
        url: The URL of the webpage to scrape.
        skip_ssl_verify: If True, disables SSL certificate verification.

    Returns:
        The extracted text content from the page.
    """
    # Try Firecrawl first if configured (skip placeholder keys)
    if settings.firecrawl_api_key and not settings.firecrawl_api_key.startswith("fc-your"):
        try:
            from firecrawl import FirecrawlApp
            app = FirecrawlApp(api_key=settings.firecrawl_api_key)
            result = app.scrape_url(url)

            if isinstance(result, dict):
                content = result.get("markdown", result.get("content", "No content extracted."))
            else:
                content = str(result)

            max_len = 8000
            if len(content) > max_len:
                content = content[:max_len] + "\n\n... [content truncated]"

            return content
        except Exception as e:
            logger.warning(f"Firecrawl failed ({e}), falling back to httpx scraper")

    # Fallback to httpx + BeautifulSoup (free)
    return _httpx_scrape(url, skip_ssl_verify=skip_ssl_verify)


# Collect all tools for easy import
research_tools = [web_search]
scraper_tools = [scrape_webpage]
all_tools = [web_search, scrape_webpage]
