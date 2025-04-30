#!/usr/bin/env python
"""
One-shot script: build local BM25 + FAISS indexes from the Chinese-cuisine PDF.
Run **once**, or whenever the PDF changes.
"""

from pathlib import Path
from langchain.schema import Document
import os

from tools.rag.pdf_parse import DataProcess
from tools.rag.bm25_retriever import BM25
from tools.rag.faiss_retriever import FaissRetriever

ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT / "data" / "how_to_cook.pdf"
OUT = Path("./indexes")          # folder to keep artifacts
OUT.mkdir(parents=True, exist_ok=True)

bm25_path = OUT / "bm25.pkl"
faiss_path = OUT / "faiss"

# ---- check if index exists ------------------------------------------------
if bm25_path.exists() and faiss_path.exists():
    print("✅  Indexes already exist, skipping rebuild.")
else:
    dp = DataProcess(PDF_PATH)
    dp.parse(max_seq=512)
    texts = dp.data

    # ---- BM25 -------------------------------------------------------
    bm25 = BM25(texts)
    bm25.save(bm25_path)

    # ---- FAISS ------------------------------------------------------
    faiss = FaissRetriever(texts, model_name="text-embedding-3-large", chunk_size=128)
    faiss.save(faiss_path)

    print("✅  Indexes built and saved to", OUT.resolve())
