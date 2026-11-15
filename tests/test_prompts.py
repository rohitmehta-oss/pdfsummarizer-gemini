from __future__ import annotations

from prompts import ANALYSIS, CHUNK_SUMMARY, COMBINE, QUOTES, STYLES, SYSTEM


def test_styles_cover_supported_summary_types() -> None:
    assert set(STYLES) == {"brief", "detailed", "bullet_points", "executive"}


def test_prompt_variables() -> None:
    assert set(CHUNK_SUMMARY.input_variables) == {"style", "text"}
    assert set(COMBINE.input_variables) == {"style", "summaries"}
    assert set(ANALYSIS.input_variables) == {"text"}
    assert set(QUOTES.input_variables) == {"text"}


def test_chunk_prompt_formats_system_and_human_messages() -> None:
    messages = CHUNK_SUMMARY.format_messages(style="brief", text="A short document.")

    assert len(messages) == 2
    assert messages[0].content == SYSTEM
    assert "brief" in messages[1].content
    assert "A short document." in messages[1].content


def test_combine_prompt_includes_style_and_sections() -> None:
    messages = COMBINE.format_messages(style="detailed", summaries="Section 1: Alpha")

    assert "detailed" in messages[1].content
    assert "Section 1: Alpha" in messages[1].content
