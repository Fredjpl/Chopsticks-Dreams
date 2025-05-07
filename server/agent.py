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
    hist_for_gk = "\n".join(f"U:{u} A:{a}" for u, a in mem.history[-5:])
    rag_needed, _ = await need_rag(hist_for_gk, msg, sid)

    # 2. context selection
    cached_ctx = None if rag_needed or mem.last_rag is None else mem.last_rag

    # 3. main agent (stick the last topic in front so the LLM “knows the it”)
    user_query = f"{mem.last_topic or ''} {msg}".strip() if not rag_needed else msg
    answer, ctx = await chef_agent.answer_query(user_query,
                                                history_pairs,
                                                cached_ctx)
    # 4. update memory
    mem.add_interaction(msg, answer)

    if rag_needed:
        topic = chef_agent.detect_topic(msg, ctx) or mem.last_topic or ""
        mem.last_topic = topic
        mem.last_rag   = chef_agent.filter_passages(topic, ctx)
        prefix = mem.last_topic or ""
        if len(prefix) > 40:
            prefix = ""                     # too long – skip
        user_query = f"{prefix} {msg}".strip() if not rag_needed else msg

    return answer

def get_response(msg: str, session_id: str | None = None) -> str:
    return asyncio.run(_async_get_response(msg, session_id))
