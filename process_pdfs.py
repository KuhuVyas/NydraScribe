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

def build_feature_matrix(spans: List[Span], median_size: float):
    """Transform Span list into ∣spans∣×8 feature numpy array."""
    feats = np.zeros((len(spans), 8), dtype=np.float32)
    for i, sp in enumerate(spans):
        feats[i, 0] = sp.font_size / median_size
        feats[i, 1] = 1.0 if sp.bold else 0.0
        feats[i, 2] = sp.y0 / 800  # normalised Y
        feats[i, 3] = sp.x0 / 600  # normalised X
        feats[i, 4] = len(sp.text.split())        # token count
        feats[i, 5] = 1.0 if sp.text.isupper() else 0.0
        feats[i, 6] = 1.0 if re.match(r"^[\\dIVXLCDM]", sp.text) else 0.0
        feats[i, 7] = 0.0                         # placeholder OCR conf
    return feats

def label_to_level(lbl: int) -> str:
    mapping = {0: "BODY", 1: "H4", 2: "H3", 3: "H2", 4: "H1", 5: "TITLE"}
    return mapping.get(lbl, "BODY")

def process_pdf(pdf_path: Path, clf: DecisionTreeClassifier, schema):
    doc = fitz.open(pdf_path)
    outline = []
    title = ""
    for pn, page in enumerate(doc, start=1):
        spans = extract_spans(page, pn)
        if not spans:
            continue
        median_size = np.median([sp.font_size for sp in spans])
        cand_spans = basic_heuristic_filter(spans, median_size)
        if not cand_spans:
            continue

        X = build_feature_matrix(cand_spans, median_size)
        preds = clf.predict(X)
        for sp, lbl in zip(cand_spans, preds):
            lvl_tag = label_to_level(lbl)
            if lvl_tag in {"H1", "H2", "H3", "H4"}:
                outline.append({"level": lvl_tag, "text": sp.text, "page": sp.page_num})
            elif lvl_tag == "TITLE" and not title:
                title = sp.text
    if not title and outline:
        title = outline[0]["text"]

    result = {"title": title, "outline": outline}
    try:
        validate(result, schema)
    except ValidationError as err:
        print(f"Schema validation failed for {pdf_path.name}: {err}", file=sys.stderr)
        return None

    return result

def main():
    schema = load_schema()
    clf     = joblib.load(MODEL_PATH)





