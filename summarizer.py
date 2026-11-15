from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, STYLES

load_dotenv()


class SummarizerError(Exception):
    pass


def _message_text(message: object) -> str:
    text = getattr(message, "text", None)
    if text is not None:
        return str(text)
    return str(getattr(message, "content", message))


_TEXT_PARSER = RunnableLambda(_message_text)


def _style(summary_type: str) -> str:
    try:
        return STYLES[summary_type]
    except KeyError as exc:
        raise ValueError(f"Unknown summary type: {summary_type}") from exc


class PDFSummarizer:
    _QUOTE_PREFIX = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s*")
    _QUOTE_CHARS = " \"'“”‘’"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        llm: Runnable | None = None,
        max_attempts: int = 3,
        max_concurrency: int = 4,
        retry_jitter: bool = True,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        if llm is None:
            key = api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                raise SummarizerError("GEMINI_API_KEY is not set")
            llm = ChatGoogleGenerativeAI(
                model=model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
                temperature=0.3,
                google_api_key=key,
            )

        self.llm = llm
        self.max_concurrency = max_concurrency
        retry = {
            "stop_after_attempt": max_attempts,
            "wait_exponential_jitter": retry_jitter,
        }

        def build(prompt):
            chain = prompt | self.llm | _TEXT_PARSER
            return chain if max_attempts == 1 else chain.with_retry(**retry)

        self.chunk_chain = build(CHUNK_SUMMARY)
        self.combine_chain = build(COMBINE)
        self.analysis_chain = build(ANALYSIS)
        self.quotes_chain = build(QUOTES)

    def summarize_text(self, text: str, summary_type: str = "detailed") -> str:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        return self.chunk_chain.invoke({"style": _style(summary_type), "text": text})

    def summarize_chunks(
        self, chunks: list[str], summary_type: str = "detailed"
    ) -> dict[str, object]:
        if not chunks or any(not chunk.strip() for chunk in chunks):
            raise ValueError("chunks must contain non-empty text")

        style = _style(summary_type)
        inputs = [{"style": style, "text": chunk} for chunk in chunks]
        results = self.chunk_chain.batch(
            inputs,
            config={"max_concurrency": self.max_concurrency},
            return_exceptions=True,
        )

        individual = []
        for index, (chunk, result) in enumerate(zip(chunks, results, strict=True), start=1):
            ok = not isinstance(result, BaseException)
            summary = result if ok else f"Error processing chunk: {result}"
            individual.append(
                {
                    "chunk_number": index,
                    "summary": summary,
                    "ok": ok,
                    "original_length": len(chunk),
                    "summary_length": len(summary) if ok else 0,
                }
            )

        return {
            "individual_summaries": individual,
            "combined_summary": self.combine_summaries(individual, summary_type),
            "total_chunks": len(chunks),
            "summary_type": summary_type,
        }

    def combine_summaries(
        self, chunk_summaries: list[dict[str, object]], summary_type: str
    ) -> str:
        valid = [str(item["summary"]) for item in chunk_summaries if item.get("ok")]
        if not valid:
            return "No valid summaries were generated."
        if len(valid) == 1:
            return valid[0]

        sections = "\n\n".join(
            f"Section {index}: {summary}" for index, summary in enumerate(valid, start=1)
        )
        try:
            return self.combine_chain.invoke(
                {"style": _style(summary_type), "summaries": sections}
            )
        except Exception as exc:
            return f"Combined Summary:\n\n{sections}\n\n(Combine step failed: {exc})"

    def analyze_document_structure(
        self, text: str, max_chars: int = 3000
    ) -> dict[str, str]:
        if not text or not text.strip():
            return {"analysis": "No text to analyze.", "status": "error"}
        try:
            analysis = self.analysis_chain.invoke({"text": text[:max_chars]})
            return {"analysis": analysis, "status": "success"}
        except Exception as exc:
            return {"analysis": f"Error analyzing document: {exc}", "status": "error"}

    def extract_key_quotes(
        self, text: str, max_chars: int = 5000, limit: int = 10
    ) -> list[str]:
        if not text or not text.strip() or limit < 1:
            return []

        response = self.quotes_chain.invoke({"text": text[:max_chars]})
        quotes = []
        seen = set()
        for line in response.splitlines():
            quote = self._QUOTE_PREFIX.sub("", line).strip().strip(self._QUOTE_CHARS)
            if not quote or quote in seen:
                continue
            seen.add(quote)
            quotes.append(quote)
            if len(quotes) >= limit:
                break
        return quotes
