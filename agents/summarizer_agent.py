from autogen import AssistantAgent
from ..tools.summarizer_tool import summarize_answer
from ..config import OPENAI_MODEL

summarizer_agent = AssistantAgent(
    name="Summarizer_Agent",
    llm_config={"model": OPENAI_MODEL, "temperature": 0.3},
    tools=[summarize_answer],
)
