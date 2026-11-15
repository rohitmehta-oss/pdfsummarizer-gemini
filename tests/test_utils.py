from __future__ import annotations

from utils import (
    SUMMARY_COLUMNS,
    api_key_error,
    create_summary_dataframe,
    estimate_processing_time,
    export_summary_to_text,
)

SUMMARY_DATA = {
    "individual_summaries": [
        {
            "chunk_number": 1,
            "summary": "First summary",
            "ok": True,
            "original_length": 100,
            "summary_length": 20,
        },
        {
            "chunk_number": 2,
            "summary": "Error processing chunk: timeout",
            "ok": False,
            "original_length": 50,
            "summary_length": 0,
        },
    ],
    "combined_summary": "Combined summary",
    "total_chunks": 2,
    "summary_type": "detailed",
}


def test_api_key_error_accepts_a_realistic_key():
    assert api_key_error("AQ." + "a" * 50) is None


def test_api_key_error_rejects_missing_key():
    message = api_key_error("")

    assert message is not None
    assert "GEMINI_API_KEY" in message


def test_api_key_error_rejects_short_key():
    message = api_key_error("short")

    assert message is not None
    assert "incomplete" in message


def test_create_summary_dataframe_has_expected_values():
    frame = create_summary_dataframe(SUMMARY_DATA)

    assert list(frame.columns) == SUMMARY_COLUMNS
    assert frame.loc[0, "Compression Ratio"] == 0.2
    assert frame.loc[0, "Status"] == "Success"
    assert frame.loc[1, "Compression Ratio"] == 0.0
    assert frame.loc[1, "Status"] == "Error"


def test_create_summary_dataframe_handles_empty_results():
    frame = create_summary_dataframe({"individual_summaries": []})

    assert frame.empty
    assert list(frame.columns) == SUMMARY_COLUMNS


def test_export_summary_to_text_includes_every_section():
    report = export_summary_to_text(SUMMARY_DATA, "paper.pdf")

    assert "File: paper.pdf" in report
    assert "Summary Type: Detailed" in report
    assert "FINAL SUMMARY" in report
    assert "Combined summary" in report
    assert "Chunk 1:" in report
    assert "Chunk 2 (FAILED):" in report


def test_estimate_processing_time_formats_seconds_and_minutes():
    assert estimate_processing_time(0) == "~5 seconds"
    assert estimate_processing_time(4) == "~8 seconds"
    assert estimate_processing_time(100) == "~1 min 20s"
