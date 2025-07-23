import json
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List

import fitz  # pymupdf

from ml_classifier import HeadingClassifier
from feature_extraction import extract_spans, spans_to_features, assign_heading_levels
from utils import load_json_schema, validate_output_json

SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "output_schema.json"
INPUT_DIR = Path(__file__).parent.parent / "input"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
MODEL_PATH = Path(__file__).parent.parent / "assets" / "heading_dt.pkl"


def label_to_level(lbl):
    """Convert numeric class label to heading level string."""
    mapping = {
        0: "BODY",
        1: "H4",
        2: "H3",
        3: "H2",
        4: "H1",
        5: "TITLE",
    }
    return mapping.get(lbl, "BODY")


def filter_dense_lists(outline: List[dict]) -> List[dict]:
    """Remove consecutive headings of same level that are likely bullet lists"""
    if len(outline) < 6:
        return outline

    filtered = []
    i = 0

    while i < len(outline):
        current_level = outline[i]["level"]
        consecutive_count = 1

        # Count consecutive items of same level
        j = i + 1
        while j < len(outline) and outline[j]["level"] == current_level:
            consecutive_count += 1
            j += 1

        # If more than 5 consecutive items of same level, likely a bullet list
        if consecutive_count > 5:
            # Keep only the first one as a heading, skip the rest
            filtered.append(outline[i])
            i = j
        else:
            # Keep all items in this group
            filtered.extend(outline[i:j])
            i = j

    return filtered


def process_pdf(pdf_path, clf, schema):
    doc = fitz.open(pdf_path)
    title = ""
    all_spans = []
    all_predictions = []

    # Process all pages first
    for page_num, page in enumerate(doc, start=1):
        spans = extract_spans(page, page_num)
        if not spans:
            continue

        features, median_size = spans_to_features(spans)
        if features.shape[0] == 0:
            continue

        pred_labels = clf.predict(features)
        predictions = [label_to_level(lbl) for lbl in pred_labels]

        all_spans.extend(spans)
        all_predictions.extend(predictions)

    # Extract title (look for TITLE class or largest font on first page)
    for span, pred in zip(all_spans, all_predictions):
        if pred == "TITLE" and span.page_num == 1 and not title:
            title = span.text
            break

    # If no TITLE found, use largest font on first page
    if not title:
        first_page_spans = [s for s in all_spans if s.page_num == 1]
        if first_page_spans:
            largest_span = max(first_page_spans, key=lambda x: x.font_size)
            title = largest_span.text

    # Build outline with proper level assignment
    outline = assign_heading_levels(all_spans, all_predictions)

    # Filter out dense bullet lists
    outline = filter_dense_lists(outline)

    # Fallback title if still empty
    if not title and outline:
        title = outline[0]["text"]

    result = {"title": title, "outline": outline}

    if not validate_output_json(result, schema):
        print(f"Output JSON validation failed for {pdf_path}", file=sys.stderr)
        return None

    return result


def main():
    schema = load_json_schema(SCHEMA_PATH)
    clf = HeadingClassifier(MODEL_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDF files found in input directory.", file=sys.stderr)
        sys.exit(1)

    for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
        result = process_pdf(pdf_path, clf, schema)
        if result:
            out_file = OUTPUT_DIR / f"{pdf_path.stem}.json"
            out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Processing complete.")


if __name__ == "__main__":
    main()
