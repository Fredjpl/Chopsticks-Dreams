# tools/gatekeeper.py
import os, json, datetime
from pathlib import Path
import openai
import tiktoken

GK_MODEL   = os.getenv("GK_MODEL", "gpt-4o")
client     = openai.AsyncOpenAI()
enc        = tiktoken.encoding_for_model(GK_MODEL)

_PROMPT_HEADER = """\
You are a binary classifier. Decide whether the assistant must refresh
external recipe knowledge (return "RAG") or can answer from prior context
(return "NO_RAG").

Rules for RAG:
• The user presents NEW ingredients or a NEW dish name.
• The user asks to buy ingredients or wants grocery info.

Rules for NO_RAG:
• The user is still talking about the SAME dish/ingredients.
• Follow-ups like proportions, timing, videos, re-phrasing, etc.

Few-shot:
USER: I have eggs
Output: RAG

USER: I want to make Steamed Egg Custard
Output: RAG

USER: I want to watch a video tutorial for it
Output: NO_RAG

USER: I want to buy the missing ingredients
Output: RAG

Your response should exactly: RAG or NO_RAG
• Do not add any other text.
• Do not use any other format.
• Do not use any other language.
"""

async def need_rag(focus: str, new_msg: str,
                   session_id: str | None = "anon") -> tuple[bool, str]:
    """Returns (need_rag_flag, 'RAG' | 'NO_RAG')."""
    resp = await client.chat.completions.create(
        model=GK_MODEL,
        messages=[
            {"role": "system", "content": _PROMPT_HEADER},
            {"role": "user",
             "content": f"CURRENT_FOCUS: {focus}\n\nLATEST_USER:\n{new_msg}"}],
        temperature=0,
        max_tokens=3,
    )
    token = resp.choices[0].message.content.strip().upper()

    # ── debug dump ────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    Path("logs/gatekeeper").mkdir(parents=True, exist_ok=True)
    with open(Path("logs/gatekeeper") / f"{session_id}_{ts}.json",
              "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts,
                   "session": session_id,
                   "decision": token,
                   "focus": focus,
                   "user_msg": new_msg},
                  f, ensure_ascii=False, indent=2)

    return token == "RAG", token