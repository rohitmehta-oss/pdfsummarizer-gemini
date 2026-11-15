from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from pdf_processor import EmptyDocumentError, PDFProcessingError, PDFProcessor
from summarizer import PDFSummarizer
from utils import (
    api_key_error,
    create_summary_dataframe,
    estimate_processing_time,
    export_summary_to_text,
    format_summary_display,
)

load_dotenv()

MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
SUMMARY_TYPES = ["brief", "detailed", "bullet_points", "executive"]
SUMMARY_LABELS = {
    "brief": "Brief (2-3 sentences)",
    "detailed": "Detailed (comprehensive)",
    "bullet_points": "Bullet points (key ideas)",
    "executive": "Executive (decisions and actions)",
}

st.set_page_config(
    page_title="PDF Reader + AI Summarizer",
    page_icon="📄",
    layout="wide",
)


def render_sidebar() -> tuple[str, str, int, bool, bool]:
    st.sidebar.header("Settings")
    configured_model = os.getenv("GEMINI_MODEL", MODELS[0]).strip() or MODELS[0]
    models = MODELS if configured_model in MODELS else [configured_model, *MODELS]
    model = st.sidebar.selectbox(
        "Gemini model",
        models,
        index=models.index(configured_model),
        help="Pick a model your API key can access.",
    )
    summary_type = st.sidebar.selectbox(
        "Summary style",
        SUMMARY_TYPES,
        index=SUMMARY_TYPES.index("detailed"),
        format_func=lambda value: SUMMARY_LABELS[value],
    )
    max_tokens = st.sidebar.slider(
        "Max tokens per chunk",
        min_value=4000,
        max_value=10000,
        value=8000,
        step=1000,
        help="Smaller chunks use less context per request but need more calls.",
    )
    show_analysis = st.sidebar.checkbox("Analyze document structure", value=True)
    show_quotes = st.sidebar.checkbox("Extract key quotes", value=False)
    st.sidebar.divider()
    st.sidebar.caption("Built with Streamlit, LangChain, and Gemini.")
    return model, summary_type, max_tokens, show_analysis, show_quotes


def render_metadata(metadata: dict[str, int | str]) -> None:
    left, right = st.columns(2)
    left.write(f"**Pages:** {metadata.get('num_pages', 'Unknown')}")
    left.write(f"**Title:** {metadata.get('title', 'Unknown')}")
    right.write(f"**Author:** {metadata.get('author', 'Unknown')}")
    right.write(f"**Subject:** {metadata.get('subject', 'Unknown')}")


def process_pdf(
    uploaded_file: Any,
    summary_type: str,
    max_tokens: int,
    show_analysis: bool,
    show_quotes: bool,
) -> None:
    processor = st.session_state.pdf_processor
    summarizer = st.session_state.summarizer
    started = time.perf_counter()
    st.session_state.pop("results", None)

    with st.spinner("Extracting text from PDF..."):
        try:
            raw_text = processor.extract_text(uploaded_file)
        except PDFProcessingError as exc:
            st.error(f"Could not extract text: {exc}")
            return

    if not raw_text.strip():
        st.error("No extractable text found. Scanned PDFs need OCR before summarizing.")
        return
    st.success(f"Extracted {len(raw_text):,} characters")

    with st.spinner("Cleaning text and counting tokens..."):
        clean_text = processor.clean_text(raw_text)
        token_count = processor.count_tokens(clean_text)
    st.success(f"Cleaned text: {len(clean_text):,} characters, ~{token_count:,} tokens")

    with st.spinner("Splitting text into chunks..."):
        try:
            chunks = processor.chunk_text(clean_text, max_tokens)
        except (EmptyDocumentError, ValueError) as exc:
            st.error(str(exc))
            return
    st.success(f"Created {len(chunks)} chunk(s)")
    st.info(f"Estimated processing time: {estimate_processing_time(len(chunks))}")

    analysis = None
    if show_analysis:
        with st.spinner("Analyzing document structure..."):
            analysis = summarizer.analyze_document_structure(clean_text)

    with st.spinner(f"Generating {summary_type.replace('_', ' ')} summaries..."):
        try:
            summary_data = summarizer.summarize_chunks(chunks, summary_type)
        except Exception as exc:
            st.error(f"Could not generate summaries: {exc}")
            return

    quotes = None
    if show_quotes:
        with st.spinner("Extracting key quotes..."):
            try:
                quotes = summarizer.extract_key_quotes(clean_text)
            except Exception as exc:
                st.warning(f"Could not extract key quotes: {exc}")

    st.session_state.results = {
        "file_name": uploaded_file.name,
        "summary_data": summary_data,
        "analysis": analysis,
        "quotes": quotes,
        "processing_seconds": round(time.perf_counter() - started, 2),
    }
    st.success("Processing complete.")


def display_results(results: dict[str, Any]) -> None:
    analysis = results.get("analysis")
    if analysis:
        if analysis.get("status") == "success":
            st.subheader("Document Analysis")
            st.write(analysis.get("analysis", ""))
        else:
            st.warning(str(analysis.get("analysis", "Document analysis failed.")))

    quotes = results.get("quotes") or []
    if quotes:
        st.subheader("Key Quotes")
        for index, quote in enumerate(quotes, start=1):
            st.markdown(f'{index}. *"{quote}"*')

    summary_data = results["summary_data"]
    st.header("Summary Results")
    format_summary_display(summary_data)

    file_name = str(results.get("file_name", "document"))
    file_stem = Path(file_name).stem or "document"
    summary_df = create_summary_dataframe(summary_data)

    st.subheader("Export Options")
    text_col, csv_col = st.columns(2)
    text_col.download_button(
        "Download Summary as Text",
        data=export_summary_to_text(summary_data, file_name),
        file_name=f"{file_stem}_summary.txt",
        mime="text/plain",
    )
    csv_col.download_button(
        "Download Statistics as CSV",
        data=summary_df.to_csv(index=False),
        file_name=f"{file_stem}_stats.csv",
        mime="text/csv",
    )

    with st.expander("Processing Statistics"):
        st.dataframe(summary_df, hide_index=True)
        if not summary_df.empty:
            st.metric(
                "Avg Summary/Original Ratio",
                f"{summary_df['Compression Ratio'].mean():.2f}",
            )
            success_rate = (summary_df["Status"] == "Success").mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")

    elapsed = float(results.get("processing_seconds", 0))
    st.caption(f"Processed in {elapsed:.1f} seconds")


def main() -> None:
    st.title("📄 PDF Reader + AI Summarizer")
    st.markdown(
        "Upload a PDF, split it into token-aware chunks, and summarize it with "
        "Gemini and LangChain."
    )

    key_error = api_key_error(os.getenv("GEMINI_API_KEY"))
    if key_error:
        st.error(key_error)
        st.info("Get a free key at https://aistudio.google.com/apikey, then add it to `.env`.")
        st.stop()
    st.sidebar.caption("🔑 Gemini API key loaded")

    model, summary_type, max_tokens, show_analysis, show_quotes = render_sidebar()

    if "pdf_processor" not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()

    settings = (model, summary_type, max_tokens, show_analysis, show_quotes)
    if st.session_state.get("settings") != settings:
        st.session_state.pop("results", None)
        st.session_state.settings = settings

    if st.session_state.get("model") != model:
        st.session_state.summarizer = PDFSummarizer(model)
        st.session_state.model = model

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Text-based PDFs work best. Scanned PDFs need OCR first.",
    )
    file_id = (
        getattr(uploaded_file, "file_id", None) or uploaded_file.name if uploaded_file else None
    )
    if st.session_state.get("file_id") != file_id:
        st.session_state.pop("results", None)
        st.session_state.file_id = file_id

    if uploaded_file is None:
        st.info("Upload a PDF to get started.")
        return

    st.success(f"File uploaded: {uploaded_file.name}")
    try:
        metadata = st.session_state.pdf_processor.get_metadata(uploaded_file)
    except PDFProcessingError as exc:
        st.error(f"Could not read PDF metadata: {exc}")
        return
    render_metadata(metadata)

    if st.button("Process PDF", type="primary"):
        process_pdf(
            uploaded_file,
            summary_type,
            max_tokens,
            show_analysis,
            show_quotes,
        )

    if "results" in st.session_state:
        display_results(st.session_state.results)


main()
