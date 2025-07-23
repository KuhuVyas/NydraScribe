import re
from typing import List
from dataclasses import dataclass
import numpy as np
import fitz  # PyMuPDF


@dataclass
class Span:
    text: str
    font_size: float
    is_bold: bool
    x0: float
    y0: float
    page_num: int


_NUM_PATTERN = re.compile(r"^[0-9IVXLCDM]+[).]?\s*")
_BULLET_PATTERN = re.compile(r"^[•●\-–·]\s*$")  # Match bullet symbols


def extract_spans(page: fitz.Page, page_num: int) -> List[Span]:
    spans = []
    blocks = page.get_text("dict")["blocks"]
    page_width = page.rect.width

    for block in blocks:
        if block[" 0"]:
            continue
        for line in block["lines"]:
            for s in line["spans"]:
                txt = s["text"].strip()
                if not txt:
                    continue

                # Hard heuristic guard-rails: Skip obvious bullets
                if txt in {"•", "●", "-", "–", "·", "○"}:
                    continue

                # Skip very long text (unlikely to be headings)
                if len(txt.split()) > 15:
                    continue

                # Skip text ending with ':' unless font is large
                font_size = s["size"]
                if txt.endswith(":") and font_size < 14:
                    continue

                spans.append(
                    Span(
                        text=txt,
                        font_size=font_size,
                        is_bold="Bold" in s["font"],
                        x0=s["bbox"][0],
                        y0=s["bbox"][1],
                        page_num=page_num,
                    )
                )
    return spans


def spans_to_features(spans: List[Span]):
    """Convert spans to 10-dimensional feature matrix (added 2 new features)"""
    if not spans:
        return np.empty((0, 10)), 0.0

    # Calculate font size statistics
    font_sizes = [s.font_size for s in spans]
    median_font = float(np.median(font_sizes))
    p80_font = float(np.percentile(font_sizes, 80))  # 80th percentile for better heading detection

    # Calculate page dimensions for normalization
    page_width = 600.0  # Approximate page width

    feature_matrix = np.zeros((len(spans), 10), dtype=np.float32)

    for i, s in enumerate(spans):
        # Original 8 features
        feature_matrix[i, 0] = s.font_size / median_font
        feature_matrix[i, 1] = 1.0 if s.is_bold else 0.0
        feature_matrix[i, 2] = s.y0 / 800.0  # Approximate page height norm
        feature_matrix[i, 3] = s.x0 / page_width  # Page width norm
        feature_matrix[i, 4] = len(s.text.split())
        feature_matrix[i, 5] = 1.0 if s.text.isupper() else 0.0
        feature_matrix[i, 6] = 1.0 if bool(_NUM_PATTERN.match(s.text)) else 0.0
        feature_matrix[i, 7] = 1.0  # OCR confidence: 1 for native text

        # New features to improve accuracy
        feature_matrix[i, 8] = s.x0 / page_width  # indent_level (normalized indentation)
        feature_matrix[i, 9] = 1.0 if bool(_BULLET_PATTERN.match(s.text)) else 0.0  # bullet_like

    return feature_matrix, median_font


def assign_heading_levels(spans: List[Span], predictions: List[str]) -> List[dict]:
    """Assign proper heading levels based on font size clustering"""
    heading_spans = [span for span, pred in zip(spans, predictions)
                     if pred in {"H1", "H2", "H3", "H4", "TITLE"}]

    if not heading_spans:
        return []

    # Get unique font sizes from headings, sorted descending
    heading_fonts = sorted(set(span.font_size for span in heading_spans), reverse=True)

    # Create font-to-level mapping
    font_to_level = {}
    for i, font_size in enumerate(heading_fonts[:4]):  # Max 4 levels
        level_names = ["H1", "H2", "H3", "H4"]
        font_to_level[font_size] = level_names[i]

    # Build final outline
    outline = []
    for span, pred in zip(spans, predictions):
        if pred == "TITLE":
            # Keep TITLE as is for title extraction
            continue
        elif pred in {"H1", "H2", "H3", "H4"}:
            # Reassign level based on font size
            level = font_to_level.get(span.font_size, "H4")
            outline.append({
                "level": level,
                "text": span.text,
                "page": span.page_num
            })

    return outline
