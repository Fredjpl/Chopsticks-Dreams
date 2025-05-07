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
    full_hist_txt = "\n".join(f"U:{u} A:{a}" for u,a in mem.history[-10:])
    rag_required, gk_token = await need_rag(full_hist_txt, msg, session_id or "anon")


    # 2. decide context
    use_ctx = None if rag_required or mem.last_rag is None else mem.last_rag


    # 3. call main agent
    answer, ctx = await chef_agent.answer_query(msg, history_pairs, use_ctx)


    # 4. update memory
    mem.add_interaction(msg, answer)
    if rag_needed:
        mem.last_rag   = ctx
        mem.last_topic = chef_agent.detect_topic(msg, ctx)


    # 5. optional: persist for debugging
    log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
    mem.save_to_disk(os.path.join(log_dir, f"session_{session_id or 'anon'}.txt"))


    return answer


# sync wrapper for existing Flask route
def get_response(msg: str, session_id: str | None = None) -> str:
    return asyncio.run(_async_get_response(msg, session_id))
