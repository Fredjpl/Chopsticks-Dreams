"""Ask for ingredients → return suitable Chinese dishes (RAG)."""
import asyncio
from pathlib import Path
from typing import List

# Updated imports for AutoGen v0.4
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core import CancellationToken

# ---- RAG primitives --------------------------------------------
from tools.rag.pdf_parse import DataProcess
from tools.rag.bm25_retriever import BM25
from tools.rag.faiss_retriever import FaissRetriever
from tools.rag.rerank_api import APIReranker

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "how_to_cook.pdf"
INDEX_DIR = ROOT / "indexes"
BM25_PATH = INDEX_DIR / "bm25.pkl"
FAISS_PATH = INDEX_DIR / "faiss"
EMBED_MODEL = "text-embedding-3-large"          # multilingual
TOPK_LEX, TOPK_DENSE, FINAL_K = 20, 20, 6

# ----------------------------------------------------------------
# One-time startup: parse PDF + (re)build / load indexes
# ----------------------------------------------------------------
INDEX_DIR.mkdir(exist_ok=True)
dp = DataProcess(PDF_PATH)
dp.parse(max_seq=512)           # ~2-3 s
TEXTS: List[str] = dp.data

if BM25_PATH.exists():
    bm25 = BM25.load(BM25_PATH)
else:
    bm25 = BM25(TEXTS)
    bm25.save(BM25_PATH)

if FAISS_PATH.exists():
    faiss = FaissRetriever.load(FAISS_PATH, model_name=EMBED_MODEL)
else:
    faiss = FaissRetriever(TEXTS, model_name=EMBED_MODEL, chunk_size=128)
    faiss.save(FAISS_PATH)

reranker = APIReranker(model="rerank-multilingual-v3.0")

# ---------------- RAG pipeline callable -------------------------
async def ingredient_query(question: str) -> str:
    """Return a few best-matching recipe passages for *question*."""
    bm25_hits = [d.page_content for d in bm25.GetBM25TopK(question, TOPK_LEX)]
    faiss_hits = [d[0].page_content for d in faiss.GetTopK(question, TOPK_DENSE)]
    pool = {t: t for t in bm25_hits + faiss_hits}.values()
    from langchain.schema import Document
    reranked = reranker.predict(question,
                               [Document(page_content=p) for p in pool])
    top = reranked[:FINAL_K]
    return "\n\n---\n\n".join(f"{i+1}. {d.page_content}" for i, d in enumerate(top))

# ---------------- AutoGen agent wiring --------------------------
async def main():
    # Initialize the OpenAI client
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")
    
    # Create the chef assistant agent with the RAG tool
    chef_bot = AssistantAgent(
        name="ChefBot",
        system_message=(
            "You are a culinary assistant specialised in Chinese cuisine. "
            "Use the `ingredient_query` tool to fetch recipe excerpts, then "
            "suggest concrete dishes the user can cook."
            "Finish with TERMINATE when satisfied."
        ),
        model_client=model_client,
        tools=[ingredient_query],
    )
    
    # Create the user proxy agent
    user = UserProxyAgent(
        name="User", 
    )
    
    # Create a termination condition
    termination = TextMentionTermination("TERMINATE")
    
    # Set up the team chat with round-robin style
    team_chat = RoundRobinGroupChat(
        participants=[user, chef_bot],
        termination_condition=termination
    )
    
    # Run the conversation with streaming output
    task = "我有鸡蛋和西红柿，还有一点葱，我能做什么传统的中国菜？"
    await Console(team_chat.run_stream(task=task))
    
    # Clean up resources
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())