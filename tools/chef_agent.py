"""tools/recipe_rag.py — Retrieval‑Augmented Chinese‑recipe assistant

This module builds/loads the hybrid BM25 + FAISS indexes (with Cohere
cross‑encoder re‑ranker) once at import‑time, then exposes **`answer_query`** –
an *async* function that the terminal UI (``tools/ui.py``) calls to answer each
user question.

Running this file directly will drop you into the interactive chat loop that is
provided by ``tools.ui``.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import List
import datetime

import openai  # needs OPENAI_API_KEY env‑var
from pydantic import BaseModel, ConfigDict
from langchain.schema import Document

# ─────────────────────────── RAG building blocks ──────────────────────────
from tools.rag.pdf_parse import DataProcess
from tools.rag.bm25_retriever import BM25
from tools.rag.faiss_retriever import FaissRetriever
from tools.rag.rerank_api import APIReranker

# ----------------------------------------------------------------------------
# youtube_search: helper function to extract dish name from the answer
# ----------------------------------------------------------------------------
from tools.youtube_video_recommender import youtube_helper
from tools.grocery_search import grocery_helper
import re
from typing import List

# ----------------------------------------------------------------------------
# Logging (quiet by default – enable in your main app if you want)
# ----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Index / model paths & constants
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "how_to_cook.pdf"
INDEX_DIR = ROOT / "indexes"
BM25_PATH = INDEX_DIR / "bm25.pkl"
FAISS_PATH = INDEX_DIR / "faiss"

EMBED_MODEL = "text-embedding-3-large"  # multilingual – handles Chinese well
top_k_lex = 20
top_k_dense = 20
final_k = 6

# ----------------------------------------------------------------------------
# One‑off index build / load (executed at module import) – no stdout prints
# ----------------------------------------------------------------------------
INDEX_DIR.mkdir(parents=True, exist_ok=True)

if not (BM25_PATH.exists() and FAISS_PATH.exists()):
    # First run → build indexes (can take a few minutes depending on PDF size)
    logger.info("Building BM25 / FAISS indexes – first‑time setup …")
    dp = DataProcess(PDF_PATH)
    dp.parse(max_seq=512)
    texts: List[str] = dp.data
    
    bm25_tmp = BM25(texts)
    bm25_tmp.save(BM25_PATH)
    
    fr_tmp = FaissRetriever(texts, model_name=EMBED_MODEL, chunk_size=128)
    fr_tmp.save(FAISS_PATH)
    del bm25_tmp, fr_tmp  # free mem before loading normally

# Load indexes (fast) ----------------------------------------------------------------
logger.info("Loading indexes …")
bm25 = BM25.load(BM25_PATH)
faiss = FaissRetriever.load(FAISS_PATH, model_name=EMBED_MODEL)
reranker = APIReranker(model="rerank-multilingual-v3.0")

# ----------------------------------------------------------------------------
# Helper Pydantic wrapper so AutoGen / UI prints a short placeholder instead of
# the full retrieved passages when debugging.
# ----------------------------------------------------------------------------
class RagResult(BaseModel):
    content: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self) -> str:  # noqa: DunderStr: show concise preview in logs
        return "✅ RAG result (hidden)"

# ----------------------------------------------------------------------------
# Core retrieval routine (async because FAISS + Cohere calls are blocking)
# ----------------------------------------------------------------------------
async def _ingredient_query(question: str) -> RagResult:
    """Hybrid lexical + dense search followed by Cohere rerank → top passages."""
    # 1) lexical BM25
    bm25_hits = [d.page_content for d in bm25.GetBM25TopK(question, top_k_lex)]
    # 2) dense FAISS
    faiss_hits = [d[0].page_content for d in faiss.GetTopK(question, top_k_dense)]
    # 3) merge & deduplicate
    pool = {t: t for t in bm25_hits + faiss_hits}.values()
    # 4) Cohere cross‑encoder rerank (blocking ⇒ run in thread)
    loop = asyncio.get_running_loop()
    reranked = await loop.run_in_executor(
        None,
        lambda: reranker.predict(
            question, [Document(page_content=p) for p in pool]
        )[: final_k],
    )
    formatted = "\n\n---\n\n".join(
        f"{i+1}. {d.page_content}" for i, d in enumerate(reranked)
    )
    return RagResult(content=formatted)

# ----------------------------------------------------------------------------
# Async answer generator: returns final reply string for UI
# ----------------------------------------------------------------------------
async def answer_query(question: str, history: str | None = None) -> str:  # noqa: D401
    """Retrieve relevant passages & ask the LLM to craft a helpful reply."""
    logger.info("User question received: %s", question)

    # (1) Retrieve context via RAG
    rag_result = await _ingredient_query(question)
    context = rag_result.content

    # ---- SAVE the RAG passages so you can inspect them later -------------
    out_dir = Path("debug_ctx")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"context_.txt").write_text(context, encoding="utf-8")
    logger.info("Retrieved %d chars of context (saved to %s)",
                len(context), out_dir)

    # (2) Compose system / user messages for OpenAI chat completion
    prompt_messages = [
        {
            "role": "system",
            "content": (
"You are a professional Chinese culinary assistant. \nYou help users find Chinese dishes they can cook with their available ingredients.\n"
"When the user provides ingredients:\n"
"1. First use the `ingredient_query` tool to find matching or related Chinese dishes. This will return a RagResult object.\n"
"2. Then use the `get_recipe_details` tool with this RagResult object to retrieve the detailed recipe information.\n"
"3. Based on this information, suggest concrete dish names and briefly explain how the user's ingredients fit the dish.\n"
"4. If some important ingredients are missing, kindly point out the missing ingredients, and mention whether they are critical or optional.\n"
"5. If the user's input ingredients are extremely abnormal or unrelated to Chinese cooking, politely reply that the ingredients are not expected or related to the available Chinese recipes.\n"
"6. If the user shows intention or explicitly wants to **learn more / watch a tutorial** for one specific dish, append a single line at the very end (after TERMINATE) in the exact format:\n"
"   YOUTUBE_SEARCH: <dish-name-in-Chinese-or-English>\n"
"   (Example:  YOUTUBE_SEARCH: stir-fried Green Peppers and Onions)\n"
"   **You MUST output this line if and only if the intent is clear.**\n"
"7. If the user shows intention or explicitly wants to **buy the missing critical ingredients**,\n"
"   append a single line at the very end (after TERMINATE) in the exact format:\n"
"     GROCERY_SEARCH: ['item1', 'item2', ...]\n"
"   (Example:  GROCERY_SEARCH: ['green bell pepper', 'sesame oil'])\n"
"   • Only list the critical-missing items.\n"
"   • If intent is ambiguous, ask a clarifying question instead of outputting\n"
"     the line.\n"
"Always base your answers strictly on the retrieved passages. Do not hallucinate or fabricate any dishes.\n"
"End your response with TERMINATE when finished.\n"
            ),
        },
    *(
        [{
            "role": "system",
            "content": (
                "Here is the recent conversation for context "
                "(do **not** repeat verbatim; use only if it helps "
                "answer follow-up questions):\n"
                f"{history.strip()[:5000]}"      # trim if you like
            ),
        }] if history else []
    ),
        {
            "role": "user",
            "content": (
                f"The user has these ingredients / question:\n{question}\n\n"
                f"Here are relevant recipe excerpts (Chinese):\n{context}\n\n"
                "Please suggest specific Chinese dishes, explain how the given "
                "ingredients fit, and point out any missing critical vs. optional "
                "ingredients. Reply in English."
            ),
        },
    ]
    logger.info("Prompt messages prepared for OpenAI API call")
    logger.debug("Prompt messages: %s", prompt_messages)
    # (3) Call OpenAI – run sync client in executor so we remain async
    loop = asyncio.get_running_loop()

    def _openai_call():
        return openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt_messages,
            temperature=0.7,
        )

    try:
        completion = await loop.run_in_executor(None, _openai_call)
        answer = completion.choices[0].message.content.strip()
    except Exception as err:
        logger.exception("OpenAI generation failed: %s", err)
        answer = (
            "Sorry, I couldn't generate a response. "
            "Please try again later, "
            "or check your API settings.\n"
        )
    # ─────────────────────────  Post-process YOUTUBE_SEARCH  ─────────────────────────
    m = re.search(r"^YOUTUBE_SEARCH:\s*(.+)$", answer, re.MULTILINE)
    if m:
        dish_name = m.group(1).strip()
        try:
            vids = youtube_helper.search_youtube_recipes(dish_name, max_results=5)
            links = "\n".join(f"- {v['title']}: {v['url']}" for v in vids)
            replacement = (
                f"Here are some useful YouTube tutorials for **{dish_name}**:\n"
                f"{links}"
            )
            answer = re.sub(r"^YOUTUBE_SEARCH:.*$", replacement, answer, flags=re.MULTILINE)
        except Exception as e:
            logger.error("YouTube search failed: %s", e)
            # fall back to plain text notice
            answer = re.sub(
                r"^YOUTUBE_SEARCH:.*$",
                "(Sorry, I couldn't fetch video links right now.)",
                answer,
                flags=re.MULTILINE,
            )

    # ─────────────────────────  Post-process GROCERY_SEARCH  ───────────────────
    g = re.search(r'^GROCERY_SEARCH:\s*(\[[^\]]+\])', answer, re.MULTILINE)

    if g:
        # ensure double quotes before it reaches the browser
        fixed = g.group(1).replace("'", '"')
        answer = answer.replace(g.group(1), fixed)

        # leave the trigger line intact; the front-end will read it and
        # prompt for ZIP code

    #     try:
    #         vids = grocery_helper.search_grocery_store_nearby(zipcode, missing_ingredients, radius=3500)
    #         answer = re.sub(r'^GROCERY_SEARCH:', 'MISSING_INGREDIENTS:', answer,
    #                     flags=re.MULTILINE)
        # except Exception as e:
        #     logger.error("Grocery search failed: %s", e)
        #     # fall back to plain text notice
        #     answer = re.sub(
        #         r"^GROCERY_SEARCH:.*$",
        #         "(Sorry, I couldn't fetch Grocery places right now.)",
        #         answer,
        #         flags=re.MULTILINE,
        #     ) 
    return answer

# ----------------------------------------------------------------------------
# Entrypoint: start the interactive chat loop using tools.ui
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from tools import ui  # local import to avoid circular deps when ui imports us

    # Run until user types "exit" / Ctrl‑C
    asyncio.run(ui.chat_loop(answer_query))
