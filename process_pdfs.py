from pathlib import Path
import json
import re
import sys
from dataclasses import dataclass
from typing import List
import fitz                  # PyMuPDF
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
from jsonschema import validate, ValidationError
from tqdm import tqdm

SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "output_schema.json"
MODEL_PATH  = Path(__file__).resolve().parent / "assets" / "heading_dt.pkl"

@dataclass
class Span:
    text: str
    font_size: float
    bold: bool
    x0: float
    y0: float
    page_num: int

def load_schema():
    return json.loads(SCHEMA_PATH.read_text())


def header_regex():
    return re.compile(
        r"^(\d+(\.\d+)*|[IVXLCDM]+\.)?\\s*[A-Z0-9].{0,80}$"
    )

def extract_spans(page: fitz.Page, page_num: int) -> List[Span]:
    spans = []
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for s in line["spans"]:
                spans.append(
                    Span(
                        text=s["text"].strip(),
                        font_size=s["size"],
                        bold=bool("Bold" in s["font"]),
                        x0=s["bbox"][0],
                        y0=s["bbox"][1],
                        page_num=page_num,
                    )
                )
    return spans

def basic_heuristic_filter(spans: List[Span], median_size: float) -> List[Span]:
    """Fast rule-of-thumb: keep spans that COULD be headings."""
    candidates = []
    for sp in spans:
        if len(sp.text) < 2:
            continue
        large_enough = sp.font_size >= 0.9 * median_size
        looks_like_hdr = header_regex().match(sp.text.upper())
        near_top = sp.y0 < 200    # <≈ 5 cm from top of page
        if (large_enough or looks_like_hdr) and near_top:
            candidates.append(sp)
    return candidates