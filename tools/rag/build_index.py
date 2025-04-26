#!/usr/bin/env python
"""
One-shot script: build local BM25 + FAISS indexes from the Chinese-cuisine PDF.
Run **once**, or whenever the PDF changes.
"""

from pathlib import Path
from langchain.schema import Document

from tools.rag.pdf_parse import DataProcess
from tools.rag.bm25_retriever import BM25
from tools.rag.faiss_retriever import FaissRetriever

ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT / "data" / "how_to_cook.pdf"
OUT = Path("./indexes")          # folder to keep artefacts
OUT.mkdir(parents=True, exist_ok=True)

dp = DataProcess(PDF_PATH)
dp.parse(max_seq=512)
texts = dp.data

# ---- BM25 -------------------------------------------------------
bm25 = BM25(texts)
bm25.save(OUT / "bm25.pkl")

# ---- FAISS ------------------------------------------------------
faiss = FaissRetriever(texts, model_name="text-embedding-3-large",chunk_size=128)
faiss.save(OUT / "faiss")

print("✅  Indexes built in", OUT.resolve())
