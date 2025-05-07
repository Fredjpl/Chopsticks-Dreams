#!/usr/bin/env python
# coding: utf-8
"""
Parse a single Chinese-text PDF into overlapping text chunks.

Usage
-----
from tools.rag.pdf_parse import DataProcess

dp = DataProcess("data/how_to_cook.pdf")
dp.parse(max_seq=512)          # build chunks
chunks = dp.data               # list[str]
"""

from pathlib import Path
from typing import List

import pdfplumber              
from PyPDF2 import PdfReader  


class DataProcess:
    def __init__(self, pdf_path: str | Path):
        self.pdf_path = str(pdf_path)
        self.data: List[str] = []

    def _sliding_window(self, sentences: List[str],
                        kernel: int = 512, stride: int = 1) -> None:
        """
        Convert a list of sentences into fixed-length chunks
        (≤ `kernel` characters) with window stride `stride` sentences.
        Appends chunks to `self.data`.
        """
        cur, fast, slow = "", 0, 0
        while fast < len(sentences):
            sentence = sentences[fast]
            if len(cur + sentence) > kernel:
                if cur not in self.data:
                    self.data.append(cur + "。")
                # slide the window
                cur = cur[len(sentences[slow]) + 1 :]
                slow += stride
            cur += sentence + "。"
            fast += 1
        # last trailing window
        if cur and cur not in self.data:
            self.data.append(cur)

    def parse(self, max_seq: int = 512, min_len: int = 20) -> None:
        """
        Extract text from **one** PDF and populate `self.data`
        with chunks of length ≤ `max_seq` characters.

        * `min_len` filters out very short noise lines.
        * Sentences are defined by the Chinese full stop “。”.
        """
        full_text = ""

        # 1) Read all pages
        for page in PdfReader(self.pdf_path).pages:
            page_text = page.extract_text() or ""
            # 2) Basic cleaning
            cleaned_lines = []
            for line in page_text.splitlines():
                line = line.strip()
                if not line:                   
                    continue
                if line.isdigit():            
                    continue
                if "目录" in line:               
                    continue
                cleaned_lines.append(line)
            full_text += "".join(cleaned_lines) + "。"

        # 3) Sentence split + length filter
        sentences = [s.strip() for s in full_text.split("。")
                     if len(s.strip()) >= min_len]

        # 4) Sliding-window chunking
        self._sliding_window(sentences, kernel=max_seq)

    def parse_block(self, max_seq: int = 1024) -> None:
        """
        Original block parser based on font size & headers.
        Retained for backward compatibility; unused by default.
        """
        with pdfplumber.open(self.pdf_path) as pdf:
            for page_id, p in enumerate(pdf.pages):
                header = self._get_header(p)
                if header is None:
                    continue
                prev_size, seq = None, ""
                for word in p.extract_words(use_text_flow=True,
                                            extra_attrs=["size"]):
                    if word["text"].isdigit():          # skip page numbers
                        continue
                    if word["text"] in {"□", "•"}:
                        continue
                    if prev_size and abs(word["size"] - prev_size) < 1e-4:
                        seq += word["text"]
                    else:
                        if seq:
                            self._data_filter(seq, header, page_id, max_seq)
                        seq = word["text"]
                        prev_size = word["size"]
                if seq:
                    self._data_filter(seq, header, page_id, max_seq)

    def _data_filter(self, text: str, header: str,
                     page_id: int, max_seq: int = 1024) -> None:
        if len(text) < 6:
            return
        # chop long paragraph into sentences and keep reasonable ones
        if len(text) > max_seq:
            splitter = "。" if "。" in text else "\t"
            for sub in text.split(splitter):
                sub = sub.strip().replace("\n", "")
                if 5 < len(sub) < max_seq and sub not in self.data:
                    self.data.append(sub)
        else:
            text = text.replace("\n", "")
            if text not in self.data:
                self.data.append(text)

    def _get_header(self, page):
        try:
            words = page.extract_words()
        except Exception:
            return None
        for w in words:
            if "目录" in w["text"]:
                return None
            if 17 < w["top"] < 20:        # near top
                return w["text"]
        return words[0]["text"] if words else None

if __name__ == "__main__":
    pdf = Path("../../data/how_to_cook.pdf")
    dp = DataProcess(pdf)
    dp.parse(max_seq=256)
    print("chunks:", len(dp.data))
    print(dp.data[0][:120] + " …")
