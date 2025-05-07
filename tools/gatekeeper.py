# tools/gatekeeper.py
import os, json, datetime
from pathlib import Path


GK_MODEL = os.getenv("GK_MODEL", "gpt-3.5-turbo-0125")  # cheap + fast
client   = openai.AsyncOpenAI()


_PROMPT_HEADER = """\
You are a binary classifier. Decide whether the assistant must refresh
external recipe knowledge (return "RAG") or can answer from prior
context alone (return "NO_RAG").


Rules for RAG:
• The user presents NEW ingredients or a NEW dish name.
• The user asks to buy ingredients or wants grocery info.


Rules for NO_RAG:
• The user is still talking about the SAME dish/ingredients.
• Follow‑ups like proportions, timing, videos, re‑phrasing, etc.
Output exactly one token: RAG  or  NO_RAG
"""


async def need_rag(full_history: str,
                   new_msg: str,
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


    # ── debug dump ───────────────────────────────────────────────
    dbg_dir = Path("logs/gatekeeper"); dbg_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    with open(dbg_dir / f"{session_id}_{ts}.json", "w", encoding="utf‑8") as f:
        json.dump(
            {
                "timestamp": ts,
                "session":   session_id,
                "decision":  token,
                "prompt":    msg,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


    return token == "RAG", token        # (bool, raw_token)



