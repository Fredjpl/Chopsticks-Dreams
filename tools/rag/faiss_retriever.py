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


import asyncio

class FaissRetriever:
    @staticmethod
    async def _embed_batch(embeddings, chunk: List[Document]) -> List[List[float]]:
        """
        Helper to embed a batch asynchronously.
        """
        texts = [d.page_content for d in chunk]
        return await embeddings.aembed_documents(texts)

    @staticmethod
    def _batch_documents(docs: List[Document], batch_size: int) -> List[List[Document]]:
        """
        Split documents into batches of batch_size.
        """
        return [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

    @staticmethod
    def _build_index_with_progress(
        texts: Sequence[str],
        embeddings: OpenAIEmbeddings,
        batch: int = 512,
        concurrent_tasks: int = 5,
    ) -> FAISS:
        """
        Manually embed in batches with concurrency so we can show a tqdm bar,
        then build a FAISS index from the vectors.
        """

        docs = [
            Document(page_content=t.strip(), metadata={"id": i})
            for i, t in enumerate(texts) if len(t.strip()) > 4
        ]

        # 2. Split docs into batches
        doc_batches = FaissRetriever._batch_documents(docs, batch)

        # 3. Embed with asyncio
        async def embed_all_batches():
            results = []
            sem = asyncio.Semaphore(concurrent_tasks)  # limit concurrent tasks

            async def run_batch(chunk):
                async with sem:
                    return await FaissRetriever._embed_batch(embeddings, chunk)

            tasks = [run_batch(chunk) for chunk in doc_batches]

            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Embedding"):
                res = await f
                results.append(res)
            return results

        vectors_nested = asyncio.run(embed_all_batches())  # list of lists
        vectors = [vec for batch in vectors_nested for vec in batch]
        metadatas = [d.metadata for d in docs]

        # 4. Build FAISS index manually
        dim = len(vectors[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(vectors).astype("float32"))

        return FAISS(index, docs, metadatas, embeddings)

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

    def GetTopK(self, query: str, k: int = 10):
        """Return ``[(Document, score), …]`` best matches."""
        return self.vector_store.similarity_search_with_score(query, k=k)

    # helper
    def GetvectorStore(self):
        return self.vector_store
