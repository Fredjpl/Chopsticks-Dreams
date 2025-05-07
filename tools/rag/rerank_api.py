"""
API-based cross-encoder reranker (Cohere).
"""

import os
import cohere
from typing import List
from langchain.schema import Document   # so type hints resolve

__all__ = ["APIReranker"]


class APIReranker:
    """
    Wraps Cohere's /rerank endpoint.

    Default model: ``rerank-multilingual-v3.0`` — supports Chinese.
    Cost  ≈ 0.10 USD / 100 docs  (Apr-2025).
    """

    def __init__(self, model: str = "rerank-multilingual-v3.0"):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY env var not set")
        self.client = cohere.Client(api_key)
        self.model = model

    def predict(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Return the *docs* list reordered by decreasing relevance.
        """
        if not docs:
            return []

        payload = [d.page_content for d in docs]
        res = self.client.rerank(
            query=query,
            documents=payload,
            top_n=len(docs),
            model=self.model,
        )

        # Cohere → list[cohere.RerankResult]; order by score ↓
        order = [r.index for r in sorted(res.results,
                                         key=lambda r: -r.relevance_score)]
        return [docs[i] for i in order]
