"""
Research Agent: Retrieves information from the web and knowledge bases.
Directly calls search tools, scrapes top results for actual content,
and then uses the LLM to synthesize results.
"""

import logging
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import AgentState
from app.agents.tools import web_search, scrape_webpage
from app.agents.llm_factory import get_llm

logger = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """You are a Research Agent specialized in finding and synthesizing information.
You have been given web search results AND scraped content from top result pages.
Use the actual scraped content to provide a comprehensive, factual, and detailed answer.
Always cite the sources (URLs) at the end of your response.
If the search results don't contain relevant info, say so and answer from your knowledge."""

# Max number of search result URLs to scrape for actual content
MAX_URLS_TO_SCRAPE = 2


async def research_node(state: AgentState) -> dict:
    user_query = state['user_input']
    llm = get_llm(temperature=0, streaming=False)

    # Step 1: Directly call the web search tool
    sources = []
    search_results = ""
    try:
        logger.info(f"Research agent searching for: {user_query}")
        search_results = web_search.invoke({"query": user_query})
        logger.info(f"Search results length: {len(search_results)}")

        # Extract URLs from results
        for line in search_results.split("\n"):
            if line.startswith("URL:"):
                url = line.replace("URL:", "").strip()
                if url:
                    sources.append(url)
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        search_results = f"Web search failed: {str(e)}"

    # Step 2: Scrape top search result URLs for actual content
    scraped_content = ""
    if sources:
        # Filter to scrapable URLs (skip aggregators, pick article-like URLs)
        scrapable_urls = [
            u for u in sources
            if not any(skip in u for skip in ["news.google.com", "youtube.com", "twitter.com", "x.com"])
        ][:MAX_URLS_TO_SCRAPE]

        for url in scrapable_urls:
            try:
                logger.info(f"Research agent scraping top result: {url}")
                content = scrape_webpage.invoke({"url": url})
                if content and len(content) > 100:
                    scraped_content += f"\n\n--- Content from {url} ---\n{content[:4000]}\n"
                    logger.info(f"Scraped {len(content)} chars from {url}")
            except Exception as e:
                logger.warning(f"Failed to scrape {url}: {e}")

    # Step 3: Use LLM to synthesize search results + scraped content
    context_parts = [f"Web Search Results:\n{search_results}"]
    if scraped_content:
        context_parts.append(f"Detailed Content from Top Results:\n{scraped_content}")
    if state.get('memory_context'):
        context_parts.append(f"Memory Context: {state['memory_context']}")

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=f"""
User Query: {user_query}

{chr(10).join(context_parts)}

Synthesize the search results and scraped content into a clear, comprehensive answer for the user.
Include specific facts, details, and source URLs. Do NOT just list links — provide actual information.
"""),
    ]

    response = await llm.ainvoke(messages)

    return {
        "messages": [response],
        "research_results": response.content,
        "sources": sources,
        "current_agent": "research",
    }
