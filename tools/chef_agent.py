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

import openai  
client = openai.AsyncOpenAI()
from pydantic import BaseModel, ConfigDict
from langchain.schema import Document
from tools.rag.pdf_parse import DataProcess
from tools.rag.bm25_retriever import BM25
from tools.rag.faiss_retriever import FaissRetriever
from tools.rag.rerank_api import APIReranker
from tools.youtube_video_recommender import youtube_helper
from tools.grocery_search import grocery_helper
import re
from typing import List

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "how_to_cook.pdf"
INDEX_DIR = ROOT / "indexes"
BM25_PATH = INDEX_DIR / "bm25.pkl"
FAISS_PATH = INDEX_DIR / "faiss"

EMBED_MODEL = "text-embedding-3-large"  # multilingual – handles Chinese well
top_k_lex = 20
top_k_dense = 20
final_k = 6

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

logger.info("Loading indexes …")
bm25 = BM25.load(BM25_PATH)
faiss = FaissRetriever.load(FAISS_PATH, model_name=EMBED_MODEL)
reranker = APIReranker(model="rerank-multilingual-v3.0")

class RagResult(BaseModel):
    content: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __str__(self) -> str:  # noqa: DunderStr: show concise preview in logs
        return "✅ RAG result (hidden)"

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

# ───────────────────────────── topic detector ────────────────────────────
TOPIC_PAT = re.compile(r"(?:recipe for|make|cook|买|做)\s+([\w\u4e00-\u9fff\s\-]+)",
                       re.I)

# TOPIC_LINE_PAT = re.compile(r"^\s*([A-Za-z\u4e00-\u9fff][^。.\n]{1,30})", re.M)

def detect_topic(user_msg: str, ctx: str) -> str | None:
    """Return a concise dish / food name or None."""
    # 1) explicit pattern in user message
    if m := TOPIC_PAT.search(user_msg):
        return m.group(1).strip().lower()

    # # 2) first heading‑like line in retrieved passages
    # if m := TOPIC_LINE_PAT.search(ctx):
    #     return m.group(1).strip().lower()

    return None

def filter_passages(topic: str, context: str) -> str:
    if not topic:
        return context
    keep = [p for p in context.split("\n\n---\n\n") if topic.lower() in p.lower()]
    return "\n\n---\n\n".join(keep) or context

async def answer_query(
    question: str,
    history: list[tuple[str, str]] | None = None, 
    precomputed_context: str | None = None,
) -> str:
    """Retrieve relevant passages & ask the LLM to craft a helpful reply."""

    # (1) Retrieve or reuse context
    if precomputed_context is not None:
        context = precomputed_context
    else:
        rag_result = await _ingredient_query(question)
        context = rag_result.content

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
"   YOUTUBE_SEARCH: <dish-name-in-English>\n"
"   Here are some kinds of intents and its corresponding examples:\n"
"   • I want to learn how to cook...\n"
"   • I want to watch a video tutorial for....\n"
"   • I want to watch the videos about...\n"
"   • I want to learn how to make...\n"
"   • I want to learn how to prepare...\n"
"   • I want to know more about...\n"
"   • I want to know how to cook...\n"
"   • I want to cook...\n"
"   • I want to learn...\n"
"   • I want to know...\n"
"   • I want to see...\n"
"   • I want to find out how to cook...\n"
"   • I want to find out how to make...\n"
"   • I want to find out how to prepare...\n"
"   • I want to find out how to learn...\n"
"   • I want to find out how to know...\n"
"   • I want to find out how to see...\n"
"   (Example:  YOUTUBE_SEARCH: stir-fried Green Peppers and Onions)\n"
"7. If the user shows explicitly wants to **buy the missing ingredients**,\n"
"   • Only list the critical-missing items.\n"
"   append a single line at the very end (after TERMINATE) in the exact format:\n"
"     GROCERY_SEARCH: ['item1', 'item2', ...]\n"
"   Here are some kinds of intents and its corresponding examples:\n"
"   • I want to buy...\n"
"   • I want to purchase...\n"
"   • I want to get...\n"
"   • I want to find...\n"
"   • I want to order...\n"
"   • I want to look for...\n"
"   • I want to find out where to buy...\n"
"   • I want to know where to buy...\n"
"   • I want to know where I can buy...\n"
"   • I want to know where I can find...\n"
"   • I want to know where I can get...\n"
"   • I want to know where I can search for...\n"
"   • I want to know where I can find out where to buy...\n"
"   • I want to know where I can find out where to purchase...\n"
"   • If intent is ambiguous, ask a clarifying question first instead of outputting the line of GROCERY_SEARCH: ['item1', 'item2', ...].\n"
"   Note: you should only output this line if the user has a strong intention to buy the missing critical ingredients.\n"
"8. **Never** output *both* `YOUTUBE_SEARCH:` and `GROCERY_SEARCH:` in the same response. \n"
"    If the user seems to want both, ask a short follow-up question so you can decide which"
"    single line to emit.\n"
"Always base your answers strictly on the retrieved passages. Do not hallucinate or fabricate any dishes.\n"
"End your response with TERMINATE when finished.\n"
            ),
        }]
    if history:
        for u_msg, b_msg in history:
            prompt_messages.append({"role": "user",      "content": u_msg})
            prompt_messages.append({"role": "assistant", "content": b_msg})
    prompt_messages.append(
        {
            "role": "user",
            "content": (
                f"The user has these ingredients / question:\n{question}\n\n"
                f"Here are relevant recipe excerpts (Chinese):\n{context}\n\n"
                "Please suggest specific Chinese dishes, explain how the given "
                "ingredients fit, and point out any missing critical vs. optional "
                "ingredients. Reply in English."
            ),
        })
    
    # (3) Call OpenAI – run sync client in executor so we remain async
    try:
        completion = await client.chat.completions.create(  
            model="gpt-4o",
            messages=prompt_messages,
            temperature=0.2,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as err:
        logger.exception("OpenAI generation failed: %s", err)
        answer = (
            "Sorry, I couldn't generate a response. "
            "Please try again later, "
            "or check your API settings.\n"
        )
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

    g = re.search(r'^GROCERY_SEARCH:\s*(\[[^\]]+\])', answer, re.MULTILINE)

    if g:
        # ensure double quotes before it reaches the browser
        fixed = g.group(1).replace("'", '"')
        answer = answer.replace(g.group(1), fixed)

    answer = re.sub(r'\bTERMINATE\b', '', answer).strip()
    return answer, context

if __name__ == "__main__":
    from tools import ui  # local import to avoid circular deps when ui imports us

    # Run until user types "exit" / Ctrl‑C
    asyncio.run(ui.chat_loop(answer_query))
