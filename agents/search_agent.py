from autogen_ext.tools.mcp import mcp_server_tools
from autogen import AssistantAgent
from ..config import SEARCH_MCP_URL, OPENAI_MODEL

_search_tools = mcp_server_tools(
    server_url=SEARCH_MCP_URL,
    description="Web search + browser tools"
)

search_agent = AssistantAgent(
    name="Search_Agent",
    llm_config={"model": OPENAI_MODEL, "temperature": 0.4},
    tools=_search_tools,
)
