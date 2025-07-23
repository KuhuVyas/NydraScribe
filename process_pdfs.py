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
