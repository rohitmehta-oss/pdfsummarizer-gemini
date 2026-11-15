from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class FakeSummarizer:
    def summarize_chunks(self, chunks, summary_type="detailed"):
        individual = [
            {
                "chunk_number": index,
                "summary": f"Summary {index}",
                "ok": True,
                "original_length": len(chunk),
                "summary_length": len(f"Summary {index}"),
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
        return {
            "individual_summaries": individual,
            "combined_summary": "Combined summary",
            "total_chunks": len(chunks),
            "summary_type": summary_type,
        }

    def analyze_document_structure(self, text):
        return {"analysis": "Analysis text", "status": "success"}

    def extract_key_quotes(self, text):
        return ["Quote one"]


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-" + "x" * 40)


def run_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=30).run(timeout=30)


def install_fake_summarizer(app: AppTest) -> None:
    app.session_state["summarizer"] = FakeSummarizer()
    app.session_state["model"] = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


def test_app_renders_settings_and_uploader():
    app = run_app()

    assert len(app.exception) == 0
    assert app.title[0].value == "📄 PDF Reader + AI Summarizer"
    assert len(app.sidebar.selectbox) == 2
    assert len(app.sidebar.slider) == 1
    assert len(app.sidebar.checkbox) == 2
    assert len(app.file_uploader) == 1
    assert any("API key loaded" in item.value for item in app.sidebar.caption)


def test_app_stops_without_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    app = run_app()

    assert len(app.exception) == 0
    assert any("GEMINI_API_KEY" in item.value for item in app.error)
    assert len(app.file_uploader) == 0
    assert len(app.sidebar.selectbox) == 0


def test_app_processes_pdf_with_fake_summarizer(pdf_bytes):
    app = run_app()
    install_fake_summarizer(app)
    app.file_uploader[0].upload("sample.pdf", pdf_bytes, "application/pdf").run(timeout=30)
    app.button[0].click().run(timeout=30)

    assert len(app.exception) == 0
    assert any("File uploaded: sample.pdf" in item.value for item in app.success)
    assert any("Summary Results" in item.value for item in app.header)
    assert any("Document Analysis" in item.value for item in app.subheader)
    assert len(app.download_button) == 2
    results = app.session_state["results"]
    assert results["summary_data"]["combined_summary"] == "Combined summary"
    assert results["processing_seconds"] >= 0


def test_app_displays_quotes_when_enabled(pdf_bytes):
    app = run_app()
    install_fake_summarizer(app)
    app.sidebar.checkbox[1].check().run(timeout=30)
    app.file_uploader[0].upload("sample.pdf", pdf_bytes, "application/pdf").run(timeout=30)
    app.button[0].click().run(timeout=30)

    assert len(app.exception) == 0
    assert any("Key Quotes" in item.value for item in app.subheader)
    assert app.session_state["results"]["quotes"] == ["Quote one"]
