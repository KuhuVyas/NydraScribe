import numpy as np
import joblib
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier


def generate_improved_dummy_data(n=1500):
    """Generate more realistic training data with better class distributions"""
    np.random.seed(42)

    # Adjusted class distribution (more BODY, fewer false headings)
    class_sizes = {
        0: int(n * 0.7),  # BODY - 70%
        1: int(n * 0.05),  # H4 - 5%
        2: int(n * 0.08),  # H3 - 8%
        3: int(n * 0.10),  # H2 - 10%
        4: int(n * 0.05),  # H1 - 5%
        5: int(n * 0.02),  # TITLE - 2%
    }

    total_samples = sum(class_sizes.values())
    X = np.zeros((total_samples, 10), dtype=np.float32)  # Updated to 10 features
    y = np.zeros(total_samples, dtype=int)

    # Improved parameters per class:
    # (font_ratio, bold_prob, y_pos_mean, x_pos_mean, token_count_mean,
    #  caps_prob, numbering_prob, ocr_conf_range, indent_mean, bullet_prob)
    params = {
        0: (1.0, 0.05, 0.5, 0.4, 12, 0.02, 0.01, (0.8, 1.0), 0.3, 0.0),  # BODY
        1: (1.15, 0.4, 0.6, 0.25, 6, 0.15, 0.40, (0.8, 1.0), 0.2, 0.0),  # H4
        2: (1.25, 0.5, 0.4, 0.15, 5, 0.25, 0.35, (0.8, 1.0), 0.1, 0.0),  # H3
        3: (1.45, 0.65, 0.3, 0.05, 4, 0.35, 0.30, (0.8, 1.0), 0.05, 0.0),  # H2
        4: (1.8, 0.75, 0.2, 0.02, 3, 0.45, 0.25, (0.8, 1.0), 0.02, 0.0),  # H1
        5: (2.2, 0.8, 0.1, 0.0, 4, 0.6, 0.1, (0.8, 1.0), 0.0, 0.0),  # TITLE
    }

    current_idx = 0
    for class_label, size in class_sizes.items():
        if size == 0:
            continue

        end_idx = current_idx + size
        fs_ratio, bold_p, y_mu, x_mu, tok_mu, caps_p, num_p, conf_range, indent_mu, bullet_p = params[class_label]

        # Generate features
        X[current_idx:end_idx, 0] = np.random.normal(fs_ratio, 0.1, size)  # font_size_ratio
        X[current_idx:end_idx, 1] = np.random.binomial(1, bold_p, size)  # is_bold
        X[current_idx:end_idx, 2] = np.random.uniform(y_mu - 0.2, y_mu + 0.3, size)  # y_pos_norm
        X[current_idx:end_idx, 3] = np.random.uniform(x_mu, x_mu + 0.2, size)  # x_indent_norm
        X[current_idx:end_idx, 4] = np.random.poisson(tok_mu, size)  # token_count
        X[current_idx:end_idx, 5] = np.random.binomial(1, caps_p, size)  # is_caps
        X[current_idx:end_idx, 6] = np.random.binomial(1, num_p, size)  # has_numbering
        X[current_idx:end_idx, 7] = np.random.uniform(*conf_range, size)  # ocr_conf
        X[current_idx:end_idx, 8] = np.random.uniform(indent_mu, indent_mu + 0.1, size)  # indent_level
        X[current_idx:end_idx, 9] = np.random.binomial(1, bullet_p, size)  # bullet_like

        y[current_idx:end_idx] = class_label
        current_idx = end_idx

    return X, y


def train_and_save_model(filepath: Path):
    """Train and save the improved heading classifier"""
    X, y = generate_improved_dummy_data()

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(X, y)

    # Save model
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, filepath)

    print(f"Improved model saved to {filepath}")
    print(f"Model size: {filepath.stat().st_size / (1024 * 1024):.3f} MB")

    # Print feature importance
    feature_names = [
        "font_size_ratio", "is_bold", "y_pos_norm", "x_indent_norm",
        "token_count", "is_caps", "has_numbering", "ocr_conf",
        "indent_level", "bullet_like"
    ]

    importance = clf.feature_importances_
    print("\nFeature Importance:")
    for name, imp in zip(feature_names, importance):
        print(f"  {name}: {imp:.3f}")


if __name__ == "__main__":
    model_path = Path("assets/heading_dt.pkl")
    train_and_save_model(model_path)
