from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "You are a precise document summarizer. Use only information in the provided "
    "text; never invent facts, numbers or names. Write in clear English. "
    "Respond with the requested content only: no introductions or closing remarks."
)

STYLES: dict[str, str] = {
    "brief": "2-3 sentences focusing on the main points and key takeaways.",
    "detailed": (
        "A detailed summary. Include:\n"
        "- Main topics and themes\n"
        "- Key arguments or findings\n"
        "- Important details and examples\n"
        "- Conclusions or recommendations"
    ),
    "bullet_points": (
        "Clear, organized bullet points. Group related information together "
        "and use sub-bullets where appropriate."
    ),
    "executive": (
        "An executive summary suitable for business leaders. Include:\n"
        "- Key insights and findings\n"
        "- Strategic implications\n"
        "- Actionable recommendations\n"
        "- Risk factors or considerations"
    ),
}


def _prompt(human: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", human)])


CHUNK_SUMMARY = _prompt(
    "Summarize the document section below in this style:\n{style}\n\n"
    "<document>\n{text}\n</document>"
)

COMBINE = _prompt(
    "Below are summaries of consecutive sections of one document. Merge them into one "
    "cohesive summary in this style:\n{style}\n\nRemove repetition.\n\n"
    "<summaries>\n{summaries}\n</summaries>"
)

ANALYSIS = _prompt(
    "Analyze the structure and content of the document below. Provide:\n\n"
    "1. Document Type\n"
    "2. Main Topics/Themes\n"
    "3. Key Sections or Chapters\n"
    "4. Target Audience\n"
    "5. Overall Purpose\n\n"
    "<document>\n{text}\n</document>"
)

QUOTES = _prompt(
    "Extract 5-10 key quotes that best represent the main ideas of the document below. "
    "Copy each quote verbatim, joining line-wrapped text into a single line. "
    "Output one quote per line, with no numbering, bullets or quotation marks.\n\n"
    "<document>\n{text}\n</document>"
)
