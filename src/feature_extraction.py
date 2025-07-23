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


def extract_spans(page: fitz.Page, page_num: int) -> List[Span]:
    spans = []
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for s in line["spans"]:
                txt = s["text"].strip()
                if not txt:
                    continue
                spans.append(
                    Span(
                        text=txt,
                        font_size=s["size"],
                        is_bold="Bold" in s["font"],
                        x0=s["bbox"][0],
                        y0=s["bbox"][1],
                        page_num=page_num,
                    )
                )
    return spans


def spans_to_features(spans: List[Span]):
    if not spans:
        return np.empty((0, 8)), 0.0
    median_font = float(np.median([s.font_size for s in spans]))
    feature_matrix = np.zeros((len(spans), 8), dtype=np.float32)
    for i, s in enumerate(spans):
        feature_matrix[i, 0] = s.font_size / median_font
        feature_matrix[i, 1] = 1.0 if s.is_bold else 0.0
        feature_matrix[i, 2] = s.y0 / 800.0  # Approximate page height norm
        feature_matrix[i, 3] = s.x0 / 600.0  # Approximate page width norm
        feature_matrix[i, 4] = len(s.text.split())
        feature_matrix[i, 5] = 1.0 if s.text.isupper() else 0.0
        feature_matrix[i, 6] = 1.0 if bool(_NUM_PATTERN.match(s.text)) else 0.0
        feature_matrix[i, 7] = 1.0  # OCR confidence: 1 for native text
    return feature_matrix, median_font
