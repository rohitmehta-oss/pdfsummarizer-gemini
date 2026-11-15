from __future__ import annotations

from io import BytesIO

import pytest

import pdf_processor
from pdf_processor import EmptyDocumentError, PDFProcessingError, PDFProcessor


@pytest.fixture(scope="module")
def processor() -> PDFProcessor:
    return PDFProcessor()


def test_extract_text_from_stream(processor: PDFProcessor, pdf_bytes: bytes) -> None:
    text = processor.extract_text(BytesIO(pdf_bytes))

    assert "Hello from the PDF" in text
    assert "This is the second sentence" in text


def test_extract_text_from_path(
    processor: PDFProcessor, pdf_bytes: bytes, tmp_path
) -> None:
    path = tmp_path / "sample.pdf"
    path.write_bytes(pdf_bytes)

    assert "Hello from the PDF" in processor.extract_text(path)


def test_extract_text_raises_for_invalid_pdf(processor: PDFProcessor) -> None:
    with pytest.raises(PDFProcessingError):
        processor.extract_text(BytesIO(b"this is not a pdf"))


def test_extract_text_skips_failed_pages(
    processor: PDFProcessor, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class FailingPage:
        def extract_text(self) -> str:
            raise RuntimeError("broken page")

    class FakeReader:
        pages = [FailingPage()]

    monkeypatch.setattr(pdf_processor, "PdfReader", lambda source: FakeReader())

    assert processor.extract_text(BytesIO(b"ignored")) == ""
    assert "Skipping page 1" in caplog.text


def test_get_metadata(processor: PDFProcessor, pdf_bytes: bytes) -> None:
    metadata = processor.get_metadata(BytesIO(pdf_bytes))

    assert metadata["num_pages"] == 1
    assert metadata["title"] == "Test Title"
    assert metadata["author"] == "Test Author"
    assert metadata["subject"] == "Test Subject"


def test_get_metadata_raises_for_invalid_pdf(processor: PDFProcessor) -> None:
    with pytest.raises(PDFProcessingError):
        processor.get_metadata(BytesIO(b"this is not a pdf"))


def test_clean_text_normalizes_whitespace_and_control_characters(
    processor: PDFProcessor,
) -> None:
    raw = "  Hello\t world\r\n\r\n\r\nNext\x00 line  "

    assert processor.clean_text(raw) == "Hello world\n\nNext line"


def test_clean_text_preserves_paragraph_breaks(processor: PDFProcessor) -> None:
    assert processor.clean_text("First paragraph.\n\nSecond paragraph.") == (
        "First paragraph.\n\nSecond paragraph."
    )


def test_count_tokens(processor: PDFProcessor) -> None:
    assert processor.count_tokens("") == 0
    assert processor.count_tokens("hello world") > 0


def test_chunk_text_rejects_empty_document(processor: PDFProcessor) -> None:
    with pytest.raises(EmptyDocumentError):
        processor.chunk_text(" \n\t ")


def test_chunk_text_rejects_invalid_limit(processor: PDFProcessor) -> None:
    with pytest.raises(ValueError):
        processor.chunk_text("Some text.", max_tokens=0)


def test_chunk_text_respects_token_limit(processor: PDFProcessor) -> None:
    text = " ".join(["word"] * 500)

    chunks = processor.chunk_text(text, max_tokens=25)

    assert len(chunks) > 1
    assert all(0 < processor.count_tokens(chunk) <= 25 for chunk in chunks)


def test_chunk_text_hard_splits_oversized_sentence(processor: PDFProcessor) -> None:
    text = "word " * 200

    chunks = processor.chunk_text(text, max_tokens=10)

    assert len(chunks) > 1
    assert all(0 < processor.count_tokens(chunk) <= 10 for chunk in chunks)


def test_chunk_text_keeps_short_document_in_one_chunk(processor: PDFProcessor) -> None:
    text = "First sentence. Second sentence. Third sentence."

    assert processor.chunk_text(text, max_tokens=1000) == [text]
