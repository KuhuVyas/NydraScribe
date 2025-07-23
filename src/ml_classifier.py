from pathlib import Path
import joblib
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "font_size_ratio",  # span font ÷ median page font
    "is_bold",  # 1/0
    "y_pos_norm",  # 0-1 (top-to-bottom)
    "x_indent_norm",  # 0-1
    "token_count",  # n words
    "is_caps",  # 1/0
    "has_numbering",  # 1/0
    "ocr_conf",  # 0-1 (1 for native text)
    "indent_level",  # NEW: normalized indentation
    "bullet_like"  # NEW: 1 if looks like bullet symbol
]

CLASS_MAP = {0: "BODY", 1: "H4", 2: "H3", 3: "H2", 4: "H1", 5: "TITLE"}
REV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


class HeadingClassifier:
    def __init__(self, model_path: Path = None):
        if model_path and model_path.exists():
            self.model = joblib.load(model_path)
        else:
            # Updated model with better parameters for heading detection
            self.model = DecisionTreeClassifier(
                criterion="gini",
                max_depth=12,  # Increased depth for better patterns
                min_samples_split=3,  # Reduced for better sensitivity
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            )

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_with_confidence(self, X, threshold=0.6):
        """Predict with confidence filtering"""
        probas = self.predict_proba(X)
        preds = self.predict(X)

        filtered_preds = []
        for i, (pred, proba) in enumerate(zip(preds, probas)):
            max_confidence = np.max(proba)
            if max_confidence >= threshold:
                filtered_preds.append(pred)
            else:
                filtered_preds.append(0)  # Default to BODY if low confidence

        return filtered_preds

    def train(self, X: np.ndarray, y: np.ndarray, save_path: Optional[Path] = None) -> Dict:
        """Train the model with evaluation"""
        x_tr, x_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.model.fit(x_tr, y_tr)
        y_pred = self.model.predict(x_te)
        acc = accuracy_score(y_te, y_pred)

        if save_path:
            self.save(save_path)

        return {
            "accuracy": acc,
            "report": classification_report(
                y_te, y_pred, target_names=list(CLASS_MAP.values()), zero_division=0
            ),
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path):
        self.model = joblib.load(path)
