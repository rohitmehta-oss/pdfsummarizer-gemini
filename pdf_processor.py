from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import BinaryIO

import tiktoken
from pypdf import PdfReader

logger = logging.getLogger(__name__)

PDFSource = str | Path | BinaryIO


class PDFProcessingError(Exception):
    pass


class EmptyDocumentError(PDFProcessingError):
    pass


class PDFProcessor:
    _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
    _CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding = tiktoken.get_encoding(encoding_name)

    @staticmethod
    def _seek(source: PDFSource) -> None:
        seek = getattr(source, "seek", None)
        if callable(seek):
            seek(0)

    def _reader(self, source: PDFSource) -> PdfReader:
        self._seek(source)
        return PdfReader(source)

    def extract_text(self, source: PDFSource) -> str:
        try:
            reader = self._reader(source)
            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    pages.append(page.extract_text() or "")
                except Exception as exc:
                    logger.warning("Skipping page %d: %s", page_number, exc)
        except Exception as exc:
            raise PDFProcessingError(f"Unable to read PDF: {exc}") from exc

        return "\n\n".join(pages).strip()

    def clean_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._CONTROL_CHARS.sub("", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def _hard_split(self, text: str, max_tokens: int) -> list[str]:
        token_ids = self.encoding.encode(text, disallowed_special=())
        return [
            self.encoding.decode(token_ids[index : index + max_tokens]).strip()
            for index in range(0, len(token_ids), max_tokens)
        ]

    def _pieces(self, text: str, max_tokens: int) -> list[str]:
        pieces = []
        for sentence in self._SENTENCE_BOUNDARY.split(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            if self.count_tokens(sentence) <= max_tokens:
                pieces.append(sentence)
            else:
                pieces.extend(self._hard_split(sentence, max_tokens))
        return pieces

    def chunk_text(self, text: str, max_tokens: int = 8000) -> list[str]:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        cleaned = self.clean_text(text)
        if not cleaned:
            raise EmptyDocumentError("No extractable text; the PDF may contain only images")

        chunks = []
        current: list[str] = []
        current_tokens = 0

        for piece in self._pieces(cleaned, max_tokens):
            piece_tokens = self.count_tokens(piece)
            separator = 1 if current else 0
            if current and current_tokens + separator + piece_tokens > max_tokens:
                chunks.append(" ".join(current))
                current = [piece]
                current_tokens = piece_tokens
            else:
                current.append(piece)
                current_tokens += separator + piece_tokens

        if current:
            chunks.append(" ".join(current))

        fitted = []
        for chunk in chunks:
            if self.count_tokens(chunk) <= max_tokens:
                fitted.append(chunk)
            else:
                fitted.extend(self._hard_split(chunk, max_tokens))

        return [chunk for chunk in fitted if chunk]

    def get_metadata(self, source: PDFSource) -> dict[str, int | str]:
        try:
            reader = self._reader(source)
            metadata = reader.metadata or {}
            num_pages = len(reader.pages)
        except Exception as exc:
            raise PDFProcessingError(f"Unable to read PDF metadata: {exc}") from exc

        def value(key: str) -> str:
            item = metadata.get(key)
            return str(item) if item else "Unknown"

        return {
            "num_pages": num_pages,
            "title": value("/Title"),
            "author": value("/Author"),
            "subject": value("/Subject"),
        }
