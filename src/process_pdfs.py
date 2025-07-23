import json
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List
import re

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


def multi_stage_heading_filter(spans, predictions, clf):
    """Apply multiple filters to reduce false positives"""

    # Get confidence scores
    try:
        features, _ = spans_to_features(spans)
        if hasattr(clf.model, 'predict_proba'):
            confidence_scores = np.max(clf.model.predict_proba(features), axis=1)
        else:
            confidence_scores = [1.0] * len(predictions)
    except:
        confidence_scores = [1.0] * len(predictions)

    filtered_results = []
    for span, pred, conf in zip(spans, predictions, confidence_scores):

        # Rule 1: Skip very long text (body paragraphs)
        if len(span.text.split()) > 20:
            filtered_results.append("BODY")
            continue

        # Rule 2: Skip text ending with common body text patterns
        if span.text.endswith(('.', ',', ';', 'years', 'experience')):
            filtered_results.append("BODY")
            continue

        # Rule 3: Skip text starting with lowercase (likely continuation)
        if span.text and span.text[0].islower():
            filtered_results.append("BODY")
            continue

        # Rule 4: Skip very short meaningless text
        if len(span.text.strip()) < 2:
            filtered_results.append("BODY")
            continue

        # Rule 5: Require minimum confidence for heading classification
        if pred != "BODY" and conf < 0.65:
            filtered_results.append("BODY")
            continue

        filtered_results.append(pred)

    return filtered_results


def assign_contextual_heading_levels(spans: List, predictions: List[str]) -> List[dict]:
    """Assign heading levels based on document context and font hierarchy"""

    # Filter to get only heading spans
    heading_data = []
    for span, pred in zip(spans, predictions):
        if pred in {"H1", "H2", "H3", "H4", "TITLE"}:
            heading_data.append({
                'span': span,
                'font_size': span.font_size,
                'text': span.text,
                'page': span.page_num,
                'prediction': pred
            })

    if not heading_data:
        return []

    # Extract font sizes and sort by size (largest first)
    font_sizes = [h['font_size'] for h in heading_data]
    unique_sizes = sorted(set(font_sizes), reverse=True)

    # Create size-to-level mapping
    level_mapping = {}
    for i, size in enumerate(unique_sizes[:4]):  # Max 4 heading levels
        level_mapping[size] = f"H{i + 1}"

    # Apply contextual rules
    final_headings = []
    for heading in heading_data:
        # Base level from font size
        base_level = level_mapping.get(heading['font_size'], "H4")

        # Contextual adjustments
        text_lower = heading['text'].lower()

        # Promote important sections to H1
        if any(keyword in text_lower for keyword in ['pathway options', 'main', 'overview', 'introduction']):
            final_level = "H1"
        # Demote obvious subsections
        elif text_lower.endswith(':') and len(heading['text']) < 30:
            final_level = "H3"
        # Promote all-caps headers that are reasonably sized
        elif heading['text'].isupper() and 5 <= len(heading['text'].split()) <= 10:
            final_level = "H2"
        else:
            final_level = base_level

        final_headings.append({
            'level': final_level,
            'text': heading['text'],
            'page': heading['page']
        })

    return final_headings


def extract_title_intelligently(all_spans, all_predictions):
    """Extract title using multiple strategies"""
    title = ""

    # Strategy 1: Look for TITLE class prediction
    for span, pred in zip(all_spans, all_predictions):
        if pred == "TITLE" and span.page_num == 1 and not title:
            if len(span.text.split()) >= 2:  # Meaningful title
                title = span.text
                break

    # Strategy 2: Use largest font on first page
    if not title:
        first_page_spans = [s for s in all_spans if s.page_num == 1]
        if first_page_spans:
            # Sort by font size, then by position (top first)
            sorted_spans = sorted(first_page_spans,
                                  key=lambda x: (x.font_size, -x.y0), reverse=True)
            for span in sorted_spans:
                if len(span.text.split()) >= 2 and len(span.text) < 100:
                    title = span.text
                    break

    # Strategy 3: Look for text at very top of first page
    if not title:
        first_page_spans = [s for s in all_spans if s.page_num == 1]
        top_spans = [s for s in first_page_spans if s.y0 < 100]  # Top 100 points
        if top_spans:
            largest_top = max(top_spans, key=lambda x: x.font_size)
            if len(largest_top.text.split()) >= 2:
                title = largest_top.text

    return title


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


def remove_duplicates_and_cleanup(outline: List[dict]) -> List[dict]:
    """Remove duplicate headings and clean up the outline"""
    seen = set()
    cleaned = []

    for item in outline:
        # Create a key based on text and level
        key = (item['text'].strip().lower(), item['level'])

        if key not in seen and len(item['text'].strip()) > 1:
            seen.add(key)
            cleaned.append(item)

    return cleaned


def process_pdf(pdf_path, clf, schema):
    """Process a single PDF file and extract its outline"""
    doc = None
    try:
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

            # Use confidence threshold prediction if available
            if hasattr(clf, 'predict_with_confidence_threshold'):
                pred_labels = clf.predict_with_confidence_threshold(features, confidence_threshold=0.70)
            else:
                pred_labels = clf.predict(features)

            predictions = [label_to_level(lbl) for lbl in pred_labels]

            # Apply multi-stage filtering
            predictions = multi_stage_heading_filter(spans, predictions, clf)

            all_spans.extend(spans)
            all_predictions.extend(predictions)

        # Extract title intelligently
        title = extract_title_intelligently(all_spans, all_predictions)

        # Build outline with contextual level assignment
        outline = assign_contextual_heading_levels(all_spans, all_predictions)

        # Filter out dense bullet lists
        outline = filter_dense_lists(outline)

        # Remove duplicates and cleanup
        outline = remove_duplicates_and_cleanup(outline)

        # Fallback title if still empty
        if not title and outline:
            title = outline[0]["text"]

        # Final fallback for title
        if not title:
            title = "Untitled Document"

        result = {"title": title, "outline": outline}

        # Validate against schema
        if not validate_output_json(result, schema):
            print(f"Output JSON validation failed for {pdf_path}", file=sys.stderr)
            return None

        return result

    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}", file=sys.stderr)
        return None
    finally:
        if doc is not None:
            doc.close()


def main():
    """Main function to process all PDFs in input directory"""
    try:
        schema = load_json_schema(SCHEMA_PATH)
        clf = HeadingClassifier(MODEL_PATH)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        pdfs = sorted(p for p in INPUT_DIR.glob("*.pdf"))
        if not pdfs:
            print("No PDF files found in input directory.", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(pdfs)} PDF files to process.")

        for pdf_path in tqdm(pdfs, desc="Processing PDFs"):
            result = process_pdf(pdf_path, clf, schema)
            if result:
                out_file = OUTPUT_DIR / f"{pdf_path.stem}.json"
                out_file.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                print(f"Successfully processed: {pdf_path.name}")
            else:
                print(f"Failed to process: {pdf_path.name}")

        print("Processing complete.")

    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()