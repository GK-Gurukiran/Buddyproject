
"""
LangGraph workflow definition.
Builds the multi-agent graph with Supervisor routing to Research and Scraper agents.
"""

import re

from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.research_agent import research_node
from app.agents.scraper_agent import scraper_node
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.llm_factory import get_llm


RESPONSE_ASSEMBLER_PROMPT = """You are a Response Assembler. Take the results from specialist agents
and compose a final, coherent response for the user. Be clear, helpful, and conversational.
Remove redundancy between different agent outputs. Include sources/URLs at the end if available."""


async def response_assembler_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.3, streaming=False)


    parts = []
    agent_trace = []
    user_input = state.get("user_input", "")
    if state.get("research_results"):
        parts.append(f"Research Agent Findings:\n{state['research_results']}")
        agent_trace.append("research")
    if state.get("scrape_results"):
        parts.append(f"Scraper Agent Summary:\n{state['scrape_results']}")
        agent_trace.append("scraper")
    # Force 'scraper' in agent trace if user query contains a URL
    # Always include 'scraper' in agent trace if user query contains a URL
    if re.search(r'https?://|www\.', user_input):
        if "scraper" not in agent_trace:
            agent_trace.append("scraper")

    if not parts:
        return {"final_response": "I wasn't able to find relevant information for your query."}

    messages = [
        SystemMessage(content=RESPONSE_ASSEMBLER_PROMPT),
        HumanMessage(content=f"""
User's Original Question: {state['user_input']}

Agent Results:
{chr(10).join(parts)}

Sources: {', '.join(state.get('sources', []))}

Agents involved: {', '.join(agent_trace)}

Compose a final response for the user.
"""),
    ]

    response = await llm.ainvoke(messages)

    return {
        "messages": [response],
        "final_response": response.content,
        "current_agent": "assembler",
        "agent_trace": agent_trace,
    }

def route_supervisor(state: AgentState) -> str:
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "research":
        return "research"
    elif next_agent == "scraper":
        return "scraper"
    else:
        return "finish"


def route_after_research(state: AgentState) -> str:
    user_input = state.get("user_input", "").lower()
    if any(indicator in user_input for indicator in ["http://", "https://", "www.", ".com", ".org"]):
        return "scraper"
    return "assembler"


def build_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research", research_node)
    workflow.add_node("scraper", scraper_node)
    workflow.add_node("assembler", response_assembler_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"research": "research", "scraper": "scraper", "finish": END},
    )

    workflow.add_conditional_edges(
        "research",
        route_after_research,
        {"scraper": "scraper", "assembler": "assembler"},
    )

    workflow.add_edge("scraper", "assembler")
    workflow.add_edge("assembler", END)

    return workflow.compile()


agent_graph = build_agent_graph()
