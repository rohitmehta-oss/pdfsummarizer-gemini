from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

SUMMARY_COLUMNS = [
    "Chunk",
    "Original Length",
    "Summary Length",
    "Compression Ratio",
    "Status",
]


def api_key_error(api_key: str | None) -> str | None:
    if not api_key or not api_key.strip():
        return "GEMINI_API_KEY not found. Copy .env.example to .env and add your key."
    if len(api_key.strip()) < 20:
        return "GEMINI_API_KEY looks incomplete. Create a new key in Google AI Studio."
    return None


def estimate_processing_time(num_chunks: int) -> str:
    seconds = max(5, int(num_chunks * 3 / 4) + 5)
    if seconds < 60:
        return f"~{seconds} seconds"
    minutes, remainder = divmod(seconds, 60)
    if remainder:
        return f"~{minutes} min {remainder}s"
    return f"~{minutes} min"


def create_summary_dataframe(summary_data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in summary_data.get("individual_summaries", []):
        original = int(item.get("original_length", 0) or 0)
        summary = int(item.get("summary_length", 0) or 0)
        rows.append(
            {
                "Chunk": item.get("chunk_number", 0),
                "Original Length": original,
                "Summary Length": summary,
                "Compression Ratio": round(summary / original, 2) if original else 0.0,
                "Status": "Success" if item.get("ok") else "Error",
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def export_summary_to_text(summary_data: dict[str, Any], file_name: str = "") -> str:
    individual = summary_data.get("individual_summaries", [])
    summary_type = str(summary_data.get("summary_type", "")).replace("_", " ").title()
    lines = [
        "PDF Summary Report",
        "==================",
        "",
    ]
    if file_name:
        lines.append(f"File: {file_name}")
    lines.extend(
        [
            f"Summary Type: {summary_type}",
            f"Total Chunks: {summary_data.get('total_chunks', len(individual))}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "FINAL SUMMARY",
            "=============",
            "",
            str(summary_data.get("combined_summary", "")),
            "",
            "INDIVIDUAL CHUNK SUMMARIES",
            "==========================",
            "",
        ]
    )
    for item in individual:
        status = "" if item.get("ok") else " (FAILED)"
        lines.extend(
            [
                f"Chunk {item.get('chunk_number', '?')}{status}:",
                str(item.get("summary", "")),
                "",
            ]
        )
    return "\n".join(lines)


def format_summary_display(summary_data: dict[str, Any]) -> None:
    individual = summary_data.get("individual_summaries", [])
    successful = sum(1 for item in individual if item.get("ok"))
    summary_type = str(summary_data.get("summary_type", "")).replace("_", " ").title()

    st.subheader("Summary Statistics")
    total_col, success_col, type_col = st.columns(3)
    total_col.metric("Total Chunks", summary_data.get("total_chunks", len(individual)))
    success_col.metric("Successful Summaries", successful)
    type_col.metric("Summary Type", summary_type or "Unknown")

    st.subheader("Final Summary")
    st.write(summary_data.get("combined_summary", ""))

    with st.expander("View Individual Chunk Summaries"):
        for item in individual:
            label = f"Chunk {item.get('chunk_number', '?')}"
            if not item.get("ok"):
                label += " (failed)"
            st.write(f"**{label}**")
            st.write(item.get("summary", ""))
            st.caption(
                f"Original: {item.get('original_length', 0):,} chars | "
                f"Summary: {item.get('summary_length', 0):,} chars"
            )
            st.divider()
