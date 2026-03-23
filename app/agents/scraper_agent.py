"""
Scraper Agent: Extracts and summarizes content from web pages.
Directly calls scrape tools and then uses the LLM to summarize.
"""

import re
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import AgentState
from app.agents.tools import scrape_webpage
from app.agents.llm_factory import get_llm

logger = logging.getLogger(__name__)

SCRAPER_SYSTEM_PROMPT = """You are a Web Scraper Agent specialized in extracting and summarizing web page content.
You have been given scraped content from a web page. Extract the most relevant information
and organize it clearly for the user. Focus on what the user asked about."""


async def scraper_node(state: AgentState) -> dict:
    user_query = state['user_input']
    llm = get_llm(temperature=0, streaming=False)

    logger.info(f"Scraper agent invoked for query: {user_query}")

    skip_ssl_verify = state.get("skip_ssl_verify", False)
    # Step 1: Extract URLs from the user query
    urls = re.findall(r'https?://[^\s<>"]+', user_query)
    logger.info(f"Scraper agent found URLs: {urls}")

    sources = state.get("sources", [])
    scraped_content = ""

    # Step 2: Directly scrape each URL
    if urls:
        for url in urls[:3]:  # Max 3 URLs
            try:
                logger.info(f"Scraper agent scraping: {url} (skip_ssl_verify={skip_ssl_verify})")
                result = scrape_webpage.invoke({"url": url, "skip_ssl_verify": skip_ssl_verify})
                scraped_content += f"\n\n--- Content from {url} ---\n{result}"
                sources.append(url)
            except Exception as e:
                logger.error(f"Scrape failed for {url}: {e}")
                scraped_content += f"\n\nFailed to scrape {url}: {str(e)}"
    else:
        logger.warning("Scraper agent found no URLs to scrape.")
        scraped_content = "No URLs found in the query to scrape."

    # Include any previous research results
    if state.get('research_results'):
        scraped_content = f"Previous Research:\n{state['research_results']}\n\n{scraped_content}"

    # Step 3: Use LLM to summarize
    messages = [
        SystemMessage(content=SCRAPER_SYSTEM_PROMPT),
        HumanMessage(content=f"""
User Query: {user_query}

Scraped Content:
{scraped_content}

Summarize the relevant information for the user.
"""),
    ]
    response = await llm.ainvoke(messages)

    logger.info(f"Scraper agent finished. scrape_results length: {len(response.content)}")

    return {
        "messages": [response],
        "scrape_results": response.content,
        "sources": sources,
        "current_agent": "scraper",
    }
