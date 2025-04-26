"""
FAISS wrapper that builds / loads a vector index using **remote**
OpenAI embeddings (multilingual, handles Chinese well).
"""

import os
import torch
from pathlib import Path
from typing import List, Sequence
import faiss
from tqdm.auto import tqdm
import numpy as np

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

__all__ = ["FaissRetriever"]


class FaissRetriever:
    @staticmethod
    def _build_index_with_progress(
        texts: Sequence[str],
        embeddings: OpenAIEmbeddings,
        batch: int = 128,
    ) -> FAISS:
        """
        Manually embed in batches so we can show a tqdm bar, then
        build a FAISS index from the vectors.
        """
        # 1. Pre-create Document objects
        docs = [
            Document(page_content=t.strip(), metadata={"id": i})
            for i, t in enumerate(texts) if len(t.strip()) > 4
        ]

        # 2. Embed in batches w/ progress
        vectors, metadatas = [], []
        for i in tqdm(range(0, len(docs), batch), desc="Embedding"):
            chunk = docs[i : i + batch]
            vecs = embeddings.embed_documents([d.page_content for d in chunk])
            vectors.extend(vecs)
            metadatas.extend([d.metadata for d in chunk])

        # 3. Build FAISS index manually (one big add)
        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors).astype("float32"))

        return FAISS(index, docs, metadatas, embeddings)
    # --------------------------- build ---------------------------
    def __init__(self,
                 texts: Sequence[str],
                 model_name: str = "text-embedding-3-large",
                 chunk_size: int = 256):
        """
        *texts*  – iterable of raw strings (not Document objects).
        """
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chunk_size=chunk_size,
        )

        docs = [
            Document(page_content=t.strip(), metadata={"id": i})
            for i, t in enumerate(texts) if len(t.strip()) > 4
        ]
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        torch.cuda.empty_cache()

    # --------------------------- I/O -----------------------------
    def save(self, path: Path | str) -> None:
        """
        Persist index to *directory* ``path``.  Creates parent dirs.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(path.as_posix())

    @classmethod
    def load(cls, path: Path | str,
             model_name: str = "text-embedding-3-large") -> "FaissRetriever":
        """
        Load index previously saved by :py:meth:`save`.
        """
        embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
        vs = FAISS.load_local(Path(path).as_posix(), embeddings, 
                          allow_dangerous_deserialization=True)
        obj = object.__new__(cls)           # bypass __init__
        obj.vector_store = vs
        return obj

    # ------------------------- Retrieval -------------------------
    def GetTopK(self, query: str, k: int = 10):
        """Return ``[(Document, score), …]`` best matches."""
        return self.vector_store.similarity_search_with_score(query, k=k)

    # helper
    def GetvectorStore(self):
        return self.vector_store
