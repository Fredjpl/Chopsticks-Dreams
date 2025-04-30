"""
Planner decides the order:
1) Vision tool (MCP) → ingredients list
2) RAG_Agent to get recipe
3) Search_Agent for extra tips
4) Summarizer_Agent for concise wrap-up
"""
from autogen import AssistantAgent
from autogen_ext.tools.mcp import mcp_server_tools
from ..config import CV_MCP_URL, OPENAI_MODEL

_vision_tools = mcp_server_tools(
    server_url=CV_MCP_URL,
    description="Computer vision ingredient detector"
)

planner = AssistantAgent(
    name="Planner",
    llm_config={"model": OPENAI_MODEL, "temperature": 0.2},
    tools=_vision_tools,  # planner can call vision directly
)
