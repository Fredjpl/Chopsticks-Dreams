from autogen import AssistantAgent
from ..tools.recipe_rag import retrieve_recipe
from ..config import OPENAI_MODEL

rag_agent = AssistantAgent(
    name="RAG_Agent",
    llm_config={"model": OPENAI_MODEL, "temperature": 0.3},
    tools=[retrieve_recipe],
)
