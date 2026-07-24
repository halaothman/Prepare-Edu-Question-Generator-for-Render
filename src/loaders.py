from __future__ import annotations

import io
import os

import docx
import pdfplumber


def _format_table(table: list[list | None]) -> str:
    rows: list[list[str]] = []
    for row in table:
        if not row:
            continue
        rows.append([str(cell or "").replace("\n", " ").strip() for cell in row])
    if not rows:
        return ""
    lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _ocr_image(image) -> str:
    try:
        import pytesseract

        return pytesseract.image_to_string(image, lang="ara+eng").strip()
    except Exception:
        return ""


def _ocr_pdf_pages(path: str) -> list[str]:
    try:
        import fitz
        from PIL import Image
    except ImportError:
        return []

    if not _ocr_available():
        return []

    parts: list[str] = []
    doc = fitz.open(path)
    try:
        for page_num, page in enumerate(doc, 1):
            page_text_parts: list[str] = []

            for img_index, image_info in enumerate(page.get_images(full=True), 1):
                try:
                    extracted = doc.extract_image(image_info[0])
                    image = Image.open(io.BytesIO(extracted["image"]))
                    text = _ocr_image(image)
                    if text:
                        page_text_parts.append(f"[صورة {img_index}]\n{text}")
                except Exception:
                    continue

            if page_text_parts:
                parts.append(f"--- OCR صفحة {page_num} ---\n" + "\n\n".join(page_text_parts))
    finally:
        doc.close()

    return parts


def _ocr_scanned_page(path: str, page_index: int) -> str:
    try:
        import fitz
        from PIL import Image
    except ImportError:
        return ""

    if not _ocr_available():
        return ""

    doc = fitz.open(path)
    try:
        page = doc[page_index]
        pixmap = page.get_pixmap(dpi=200)
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        return _ocr_image(image)
    except Exception:
        return ""
    finally:
        doc.close()


def load_pdf(path: str) -> str:
    parts: list[str] = []
    sparse_pages: list[int] = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            parts.append(f"\n--- صفحة {page_num} ---\n")

            text = (page.extract_text() or "").strip()
            if text:
                parts.append(text)
            if len(text) < 50:
                sparse_pages.append(page_num - 1)

            for table_index, table in enumerate(page.extract_tables() or [], 1):
                formatted = _format_table(table)
                if formatted.strip():
                    parts.append(f"\n[جدول {table_index}]\n{formatted}\n")

    for page_index in sparse_pages:
        ocr_text = _ocr_scanned_page(path, page_index)
        if ocr_text:
            parts.append(f"\n--- OCR صفحة {page_index + 1} (مسح ضوئي) ---\n{ocr_text}\n")

    parts.extend(_ocr_pdf_pages(path))
    return "\n".join(parts).strip()


def load_docx(path: str) -> str:
    document = docx.Document(path)
    parts: list[str] = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text.strip())

    for table_index, table in enumerate(document.tables, 1):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        formatted = _format_table(rows)
        if formatted.strip():
            parts.append(f"\n[جدول {table_index}]\n{formatted}\n")

    return "\n".join(parts).strip()


def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return load_pdf(path)

    if ext in {".docx", ".doc"}:
        return load_docx(path)

    if ext in {".txt", ".md"}:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()

    raise ValueError(f"Unsupported file type: {ext}")
