import asyncio
from tools import chef_agent
from tools.ui_memory import ConversationMemory
import os
from tools.gatekeeper import need_rag

_sessions: dict[str, ConversationMemory] = {}

async def _async_get_response(msg: str, session_id: str | None):
    sid = session_id or "anon"
    mem = _sessions.setdefault(sid, ConversationMemory(5))
    history_pairs = mem.history[-5:]

    # 1. gatekeeper
    focus_txt = mem.last_topic or ""
    rag_needed, _ = await need_rag(focus_txt, msg, sid)

    # 2. context selection
    cached_ctx = None if rag_needed or mem.last_rag is None else mem.last_rag

    # 3. main agent — always add the current dish/topic to the search query
    query_for_agent = f"{mem.last_topic or ''} {msg}".strip()
    answer, ctx = await chef_agent.answer_query(query_for_agent,
                                                history_pairs,
                                                cached_ctx)
    # 4. update memory
    mem.add_interaction(msg, answer)
    if rag_needed:
        mem.last_rag   = ctx
        mem.last_topic = chef_agent.detect_topic(msg, ctx)

    # 5. persist (optional)
    os.makedirs("logs", exist_ok=True)
    mem.save_to_disk(f"logs/session_{sid}.txt")

    return answer

def get_response(msg: str, session_id: str | None = None) -> str:
    return asyncio.run(_async_get_response(msg, session_id))
