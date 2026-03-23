"""
Supervisor Agent: Orchestrates the multi-agent workflow.
Analyzes user queries and routes to the appropriate specialist agent.
"""

import json
import re
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.state import AgentState
from app.agents.llm_factory import get_llm

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Agent that orchestrates a team of AI specialist agents.

Your team members:
1. **research** - Research Agent: Searches the web for information, answers factual questions, finds current data.
2. **scraper** - Scraper Agent: Extracts and summarizes content from specific web pages/URLs.
3. **FINISH** - Return the final response directly ONLY for simple greetings or casual conversation.

Your job:
- Analyze the user's query to determine which agent(s) should handle it.
- Route to the right agent based on the query type.

ROUTING RULES (follow strictly):
- If the query mentions "search", "find", "look up", "latest", "news", "current", "today", "recent", "2024", "2025", "2026" → ALWAYS route to **research**
- If the query asks about facts, events, people, companies, technology → route to **research**
- If the query mentions a specific URL (http://, https://, www.) → route to **scraper**
- If the query needs both research and scraping → route to **research** first
- ONLY choose FINISH for simple greetings like "hi", "hello", "thanks", "bye", or trivial chat

When in doubt, route to **research**. It is better to search and provide real results than to answer from memory.

IMPORTANT: Respond with ONLY a JSON object like this:
{"next": "research"} or {"next": "scraper"} or {"next": "FINISH", "response": "your direct answer here"}

Do NOT include any other text outside the JSON.
"""

# Keywords that should ALWAYS trigger research routing
RESEARCH_KEYWORDS = [
    "search", "find", "look up", "lookup", "google", "latest", "news",
    "current", "today", "recent", "what happened", "tell me about",
    "who is", "what is", "how to", "explain", "describe", "compare",
    "difference between", "2024", "2025", "2026", "update",
]


def _should_force_research(query: str) -> bool:
    """Check if query contains keywords that should always trigger research."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in RESEARCH_KEYWORDS)


async def supervisor_node(state: AgentState) -> dict:
    user_input = state['user_input']

    # Force research for obvious search queries
    if _should_force_research(user_input):
        return {
            "messages": [],
            "current_agent": "supervisor",
            "next_agent": "research",
        }

    # Check for URLs → route to scraper
    if re.search(r'https?://|www\.', user_input):
        return {
            "messages": [],
            "current_agent": "supervisor",
            "next_agent": "scraper",
        }

    llm = get_llm(temperature=0, streaming=False)

    memory_info = ""
    if state.get("memory_context"):
        memory_info = f"\n\nRelevant memory from past conversations:\n{state['memory_context']}"

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=f"""
User Query: {user_input}
{memory_info}

Decide the next step. Respond with JSON only.
"""),
    ]

    response = await llm.ainvoke(messages)

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        decision = json.loads(content)
        next_agent = decision.get("next", "FINISH")
        direct_response = decision.get("response", "")
    except (json.JSONDecodeError, AttributeError):
        next_agent = "FINISH"
        direct_response = response.content

    result = {
        "messages": [response],
        "current_agent": "supervisor",
        "next_agent": next_agent,
    }

    if next_agent == "FINISH":
        result["final_response"] = direct_response

    return result
