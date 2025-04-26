"""
High-level hybrid retrieval pipeline:
BM25  +  dense (OpenAI)  +  Cohere cross-encoder rerank.
"""

from pathlib import Path
from typing import List

from .pdf_parse import DataProcess
from .bm25_retriever import BM25
from .faiss_retriever import FaissRetriever
from .rerank_api import APIReranker

__all__ = ["RAGPipeline"]


class RAGPipeline:
    # ------------------------------------------------------------------
    def __init__(
        self,
        bm25_index_path: Path,
        faiss_index_path: Path,
        pdf_path: Path | None = None,
        *,
        embed_model: str = "text-embedding-3-large",
        top_k_dense: int = 20,
        top_k_lex: int = 20,
        final_k: int = 6,
    ):
        # 1. Build indexes if missing
        if pdf_path and (
            not bm25_index_path.exists() or not faiss_index_path.exists()
        ):
            self._build_indexes(pdf_path, bm25_index_path,
                                faiss_index_path, embed_model)

        # 2. BM25
        if bm25_index_path.exists():
            self.bm25 = BM25.load(bm25_index_path)
        else:
            raise FileNotFoundError(
                "BM25 index not found and pdf_path not supplied")

        # 3. FAISS
        if faiss_index_path.exists():
            self.faiss = FaissRetriever.load(faiss_index_path,
                                             model_name=embed_model)
        else:
            # build from BM25 docs (fall-back safety)
            corpus = [d.page_content for d in self.bm25.full_documents]
            self.faiss = FaissRetriever(corpus, model_name=embed_model)
            self.faiss.save(faiss_index_path)

        # 4. Cohere reranker
        self.rerank = APIReranker(model="rerank-multilingual-v3.0")

        # knobs
        self.top_k_dense = top_k_dense
        self.top_k_lex = top_k_lex
        self.final_k = final_k

    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> List[str]:
        dense_docs = [d[0] for d in self.faiss.GetTopK(query, self.top_k_dense)]
        lex_docs = self.bm25.GetBM25TopK(query, self.top_k_lex)

        # deduplicate by text
        pool = {d.page_content: d for d in dense_docs + lex_docs}.values()
        reranked = self.rerank.predict(query, list(pool))
        return [d.page_content for d in reranked[: self.final_k]]

    # ------------------------------------------------------------------
    @staticmethod
    def _build_indexes(pdf_path: Path,
                       bm25_out: Path,
                       faiss_out: Path,
                       embed_model: str) -> None:
        dp = DataProcess(pdf_path)
        dp.parse(max_seq=512)
        texts = dp.data

        bm25 = BM25(texts)
        bm25.save(bm25_out)

        fr = FaissRetriever(texts, model_name=embed_model)
        fr.save(faiss_out)
