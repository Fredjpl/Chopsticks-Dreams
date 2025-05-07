#!/usr/bin/env python
# coding: utf-8


from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from .pdf_parse import DataProcess
import jieba
import pickle, gzip

class BM25(object):

    def __init__(self, documents):

        docs = []
        full_docs = []
        for idx, line in enumerate(documents):
            line = line.strip("\n").strip()
            if(len(line)<5):
                continue
            tokens = " ".join(jieba.cut_for_search(line))
            docs.append(Document(page_content=tokens, metadata={"id": idx}))
            words = line.split("\t")
            full_docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.documents = docs
        self.full_documents = full_docs
        self.retriever = self._init_bm25()

    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    def GetBM25TopK(self, query, topk):
        self.retriever.k = topk
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.invoke(query)
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans

    def save(self, path):
        with gzip.open(path, "wb") as f:
            pickle.dump(self.retriever, f)
            pickle.dump(self.full_documents, f)

    @classmethod
    def load(cls, path):
        with gzip.open(path, "rb") as f:
            retriever = pickle.load(f)
            full_docs = pickle.load(f)
        obj = object.__new__(cls)
        obj.retriever = retriever
        obj.full_documents = full_docs
        obj.documents = None      
        return obj

if __name__ == "__main__":
    pass
