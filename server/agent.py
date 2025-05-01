import asyncio
from tools import chef_agent
from tools.ui_memory import ConversationMemory

_sessions: dict[str, ConversationMemory] = {}

def get_response(msg: str, session_id: str | None = None) -> str:
    mem = _sessions.setdefault(session_id or "anon", ConversationMemory(5))
    history = mem.get_context()
    # recipe_rag.answer_query is async → run it synchronously
    answer = asyncio.run(chef_agent.answer_query(msg, history))
    mem.add_interaction(msg, answer)
    return answer