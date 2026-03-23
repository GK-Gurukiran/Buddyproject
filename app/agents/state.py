"""
LangGraph state definition.
This is the shared state object that flows through all agents in the graph.
"""

from typing import TypedDict, Annotated, List, Optional, Sequence
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """
    Shared state for the multi-agent workflow.

    Attributes:
        messages: The conversation message history (appended via operator.add)
        current_agent: Which agent is currently active
        user_input: The original user query
        research_results: Results from the Research Agent
        scrape_results: Results from the Scraper Agent
        final_response: The final assembled response
        tenant_id: Tenant context for data isolation
        user_id: User making the request
        session_id: Chat session identifier
        memory_context: Relevant memories retrieved for this query
        sources: URLs or references used in the response
        next_agent: The next agent the supervisor routes to
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    user_input: str
    research_results: Optional[str]
    scrape_results: Optional[str]
    final_response: Optional[str]
    tenant_id: str
    user_id: str
    session_id: str
    memory_context: Optional[str]
    sources: List[str]
    next_agent: Optional[str]
