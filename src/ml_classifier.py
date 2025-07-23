from pathlib import Path
import joblib
from sklearn.tree import DecisionTreeClassifier


class HeadingClassifier:
    def __init__(self, model_path: Path = None):
        if model_path and model_path.exists():
            self.model = joblib.load(model_path)
        else:
            # Untrained simple model placeholder
            self.model = DecisionTreeClassifier(max_depth=10)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: Path):
        joblib.dump(self.model, path)
