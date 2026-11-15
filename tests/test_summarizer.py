from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from summarizer import PDFSummarizer, SummarizerError


def make_llm(handler, calls: list[str] | None = None) -> RunnableLambda:
    def run(prompt) -> AIMessage:
        text = prompt.to_string()
        if calls is not None:
            calls.append(text)
        return AIMessage(content=handler(text))

    return RunnableLambda(run)


def make_summarizer(handler, **kwargs) -> PDFSummarizer:
    kwargs.setdefault("max_attempts", 1)
    kwargs.setdefault("retry_jitter", False)
    return PDFSummarizer(llm=make_llm(handler), **kwargs)


def test_summarize_text_returns_model_text() -> None:
    summarizer = make_summarizer(lambda prompt: "Short summary")

    assert summarizer.summarize_text("Some text.") == "Short summary"


def test_summarize_text_rejects_empty_text() -> None:
    summarizer = make_summarizer(lambda prompt: "unused")

    with pytest.raises(ValueError):
        summarizer.summarize_text("   ")


def test_summarize_text_rejects_unknown_style() -> None:
    summarizer = make_summarizer(lambda prompt: "unused")

    with pytest.raises(ValueError):
        summarizer.summarize_text("Some text.", summary_type="missing")


def test_summarize_chunks_maps_and_combines() -> None:
    def handler(prompt: str) -> str:
        if "Section 1" in prompt:
            return "COMBINED"
        if "first chunk" in prompt:
            return "summary one"
        if "second chunk" in prompt:
            return "summary two"
        return "unexpected"

    summarizer = make_summarizer(handler, max_concurrency=2)
    result = summarizer.summarize_chunks(["first chunk", "second chunk"])

    assert result["total_chunks"] == 2
    assert result["summary_type"] == "detailed"
    assert result["combined_summary"] == "COMBINED"
    individual = result["individual_summaries"]
    assert [item["summary"] for item in individual] == ["summary one", "summary two"]
    assert all(item["ok"] for item in individual)


def test_summarize_chunks_records_failures() -> None:
    def handler(prompt: str) -> str:
        if "bad chunk" in prompt:
            raise RuntimeError("boom")
        return "good summary"

    summarizer = make_summarizer(handler)
    result = summarizer.summarize_chunks(["good chunk", "bad chunk"])

    individual = result["individual_summaries"]
    assert individual[0]["ok"] is True
    assert individual[1]["ok"] is False
    assert "Error processing chunk" in individual[1]["summary"]
    assert individual[1]["summary_length"] == 0
    assert result["combined_summary"] == "good summary"


def test_summarize_chunks_rejects_empty_list() -> None:
    summarizer = make_summarizer(lambda prompt: "unused")

    with pytest.raises(ValueError):
        summarizer.summarize_chunks([])


def test_summarize_chunks_rejects_empty_chunk() -> None:
    summarizer = make_summarizer(lambda prompt: "unused")

    with pytest.raises(ValueError):
        summarizer.summarize_chunks(["valid", "  "])


def test_combine_summaries_returns_message_when_all_failed() -> None:
    summarizer = make_summarizer(lambda prompt: "unused")
    chunks = [{"summary": "broken", "ok": False}]

    assert summarizer.combine_summaries(chunks, "detailed") == (
        "No valid summaries were generated."
    )


def test_combine_summaries_returns_single_valid_summary() -> None:
    summarizer = make_summarizer(lambda prompt: "unused")
    chunks = [{"summary": "only summary", "ok": True}]

    assert summarizer.combine_summaries(chunks, "detailed") == "only summary"


def test_combine_summaries_uses_llm_for_multiple_sections() -> None:
    summarizer = make_summarizer(lambda prompt: "combined result")
    chunks = [
        {"summary": "first summary", "ok": True},
        {"summary": "second summary", "ok": True},
    ]

    assert summarizer.combine_summaries(chunks, "detailed") == "combined result"


def test_analyze_document_structure_success() -> None:
    summarizer = make_summarizer(lambda prompt: "Document analysis")

    result = summarizer.analyze_document_structure("Some text.")

    assert result == {"analysis": "Document analysis", "status": "success"}


def test_analyze_document_structure_handles_empty_text() -> None:
    summarizer = make_summarizer(lambda prompt: "unused")

    result = summarizer.analyze_document_structure("  ")

    assert result["status"] == "error"


def test_analyze_document_structure_handles_failure() -> None:
    def handler(prompt: str) -> str:
        raise RuntimeError("boom")

    summarizer = make_summarizer(handler)
    result = summarizer.analyze_document_structure("Some text.")

    assert result["status"] == "error"
    assert "Error analyzing document" in result["analysis"]


def test_extract_key_quotes_cleans_and_deduplicates() -> None:
    response = '1. "First quote"\n- Second quote\n• Third quote\n"First quote"'
    summarizer = make_summarizer(lambda prompt: response)

    assert summarizer.extract_key_quotes("Some text.") == [
        "First quote",
        "Second quote",
        "Third quote",
    ]


def test_extract_key_quotes_limits_results() -> None:
    response = "\n".join(f"Quote {index}" for index in range(15))
    summarizer = make_summarizer(lambda prompt: response)

    assert len(summarizer.extract_key_quotes("Some text.")) == 10


def test_extract_key_quotes_skips_empty_text() -> None:
    def handler(prompt: str) -> str:
        raise AssertionError("LLM should not be called")

    summarizer = make_summarizer(handler)

    assert summarizer.extract_key_quotes("   ") == []


def test_retry_recovers_from_transient_failure() -> None:
    calls: list[str] = []

    def handler(prompt: str) -> str:
        if len(calls) == 1:
            raise RuntimeError("transient")
        return "recovered"

    summarizer = PDFSummarizer(
        llm=make_llm(handler, calls),
        max_attempts=2,
        retry_jitter=False,
    )

    assert summarizer.summarize_text("Some text.") == "recovered"
    assert len(calls) == 2


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(SummarizerError):
        PDFSummarizer(api_key=None)
