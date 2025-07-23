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




