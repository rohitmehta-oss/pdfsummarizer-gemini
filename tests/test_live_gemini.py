from __future__ import annotations

import os

import pytest

from summarizer import PDFSummarizer

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to call the Gemini API",
)


def test_live_summarize_text() -> None:
    summarizer = PDFSummarizer()

    result = summarizer.summarize_text(
        "The sky appears blue because air scatters short wavelengths of sunlight. "
        "Water is essential for all known forms of life.",
        summary_type="brief",
    )

    assert result.strip()
