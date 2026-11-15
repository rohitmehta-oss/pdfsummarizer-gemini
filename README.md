# PDF AI Summarizer

A Streamlit + LangChain app that summarizes PDFs with Google Gemini.

Inspired by [niti007/pdfsummarizer-gemini](https://github.com/niti007/pdfsummarizer-gemini).
This project is a from-scratch implementation with additional validation and tests.

## Status

Phase 0: project scaffold, dependency setup, and Gemini/GitHub preflight validation.

## Setup

Requires Python 3.10+ and a Google Gemini API key.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# add GEMINI_API_KEY to .env
python scripts/check_setup.py
```

Get a key at <https://aistudio.google.com/apikey>.

## Run

```bash
streamlit run app.py
```

The app is implemented in the next phases.
