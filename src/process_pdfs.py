import json
import sys
from pathlib import Path
from tqdm import tqdm

import fitz  # pymupdf

from ml_classifier import HeadingClassifier
from feature_extraction import extract_spans, spans_to_features
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


def process_pdf(pdf_path, clf, schema):
    doc = fitz.open(pdf_path)
    outline = []
    title = ""

    for page_num, page in enumerate(doc, start=1):
        spans = extract_spans(page, page_num)
        if not spans:
            continue

        features, median_size = spans_to_features(spans)
        if features.shape[0] == 0:
            continue

        pred_labels = clf.predict(features)
        for sp, lbl in zip(spans, pred_labels):
            lvl = label_to_level(lbl)
            if lvl == "TITLE" and not title:
                title = sp.text
            elif lvl in {"H1", "H2", "H3", "H4"}:
                outline.append({"level": lvl, "text": sp.text, "page": sp.page_num})

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
            out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("Processing complete.")


if __name__ == "__main__":
    main()
