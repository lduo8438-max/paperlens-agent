"""
PDF structure detection using PyMuPDF.

Algorithm:
1. Extract text spans with font_size, font_flags, position
2. Compute body font size (mode of all font sizes)
3. Classify each span as header or body using thresholding + pattern matching
4. Group body text under detected headers → List[Section]
"""

import re
from collections import Counter

import fitz  # PyMuPDF
import numpy as np

from models.schemas import Section

# Known academic section keywords (case-insensitive)
SECTION_KEYWORDS = {
    "abstract", "introduction", "background", "related work",
    "methodology", "method", "methods", "approach",
    "experiment", "experiments", "evaluation", "results",
    "discussion", "conclusion", "conclusions", "references",
    "acknowledgements", "acknowledgments", "appendix",
}

# Pattern: "1. Introduction" or "1 Introduction" or "I. Introduction"
NUMBERED_PATTERN = re.compile(r"^(\d+\.?\d*|[IVX]+\.?)\s+[A-Z]")


def _get_body_font_size(doc: fitz.Document) -> float:
    """Return the most common font size in the document (body text baseline)."""
    sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    sizes.append(round(span["size"], 1))
    if not sizes:
        return 10.0
    return Counter(sizes).most_common(1)[0][0]


def _is_header(text: str, font_size: float, font_flags: int, body_size: float) -> bool:
    """Classify a span as section header using thresholding + pattern matching."""
    text = text.strip()
    if not text or len(text) > 120:
        return False

    is_bold = bool(font_flags & 16)
    is_larger = font_size >= body_size + 1.0
    keyword_match = text.lower().rstrip(".") in SECTION_KEYWORDS
    pattern_match = bool(NUMBERED_PATTERN.match(text))

    return (is_larger or is_bold) and (keyword_match or pattern_match or text.isupper())


def parse_pdf(pdf_path: str) -> list[Section]:
    """
    Parse a PDF and return a list of Sections with detected structure.
    Each Section contains: name, page, text.
    """
    doc = fitz.open(pdf_path)
    body_size = _get_body_font_size(doc)

    sections: list[Section] = []
    current_name = "Preamble"
    current_page = 1
    current_text_parts: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    if _is_header(text, span["size"], span["flags"], body_size):
                        # Save previous section
                        if current_text_parts:
                            sections.append(Section(
                                name=current_name,
                                page=current_page,
                                text=" ".join(current_text_parts).strip(),
                            ))
                        current_name = text
                        current_page = page_num
                        current_text_parts = []
                    else:
                        current_text_parts.append(text)

    # Save last section
    if current_text_parts:
        sections.append(Section(
            name=current_name,
            page=current_page,
            text=" ".join(current_text_parts).strip(),
        ))

    doc.close()
    return sections
