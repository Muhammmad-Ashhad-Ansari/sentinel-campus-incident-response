"""Three-class incident classifier for SENTINEL.

Only these five input features are used by the model:

    failed_logins, requests_per_min, unusual_hour, known_device, duplicate_count

The target ``label`` is never part of the feature matrix.  A stratified 80:20
split is used, and the model is evaluated only on the held-out test set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from .dataset import FEATURES, LABELS, load_dataset
from .validation import validate_ml_features

MODEL_NAME = "DecisionTreeClassifier"


def split_dataset(
    rows: Sequence[Dict[str, object]],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Create a reproducible stratified train/test split.

    Returns a dictionary containing the feature matrices, labels and the
    original rows for the test set (used to inspect false predictions).
    """
    if not rows:
        raise ValueError("Cannot split an empty dataset.")
    labels = [str(row["label"]) for row in rows]
    if len(set(labels)) < 2:
        raise ValueError("At least two classes are required for a stratified split.")

    index = list(range(len(rows)))
    train_idx, test_idx = train_test_split(
        index,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    # train_test_split returns the indices in a deterministic but shuffled
    # order.  Sort them for stable console output; the split itself is fixed.
    train_idx = sorted(train_idx)
    test_idx = sorted(test_idx)

    def features(row: Dict[str, object]) -> List[object]:
        return [row[name] for name in FEATURES]

    return {
        "X_train": [features(rows[i]) for i in train_idx],
        "X_test": [features(rows[i]) for i in test_idx],
        "y_train": [labels[i] for i in train_idx],
        "y_test": [labels[i] for i in test_idx],
        "train_rows": [rows[i] for i in train_idx],
        "test_rows": [rows[i] for i in test_idx],
        "train_indices": train_idx,
        "test_indices": test_idx,
        "random_state": random_state,
        "test_size": test_size,
    }


def _class_counts(labels: Sequence[str]) -> Dict[str, int]:
    return {label: sum(1 for value in labels if value == label) for label in LABELS}


def _possible_reason(row: Dict[str, object], predicted: str) -> str:
    """A cautious, human-readable hypothesis for a false prediction.

    This is not a formal explanation of the model.  It points out feature
    overlap that plausibly caused the tree to choose the wrong side of a
    boundary.  The wording is intentionally tentative.
    """
    failed = row["failed_logins"]
    rpm = row["requests_per_min"]
    duplicates = row["duplicate_count"]
    known = row["known_device"]
    unusual = row["unusual_hour"]
    return (
        f"The record has failed_logins={failed}, requests_per_min={rpm}, "
        f"unusual_hour={unusual}, known_device={known}, duplicate_count={duplicates}. "
        f"These values overlap the training ranges for {predicted}, so the "
        "decision tree may have split on a nearby threshold rather than the "
        "signal that generated the synthetic label."
    )


class IncidentClassifier:
    """Small wrapper around scikit-learn's DecisionTreeClassifier.

    The wrapper adds the five-feature contract, validation, evaluation and
    JSON-safe metrics.  It is deliberately thin so it can be explained live.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_depth: int | None = 4,
        min_samples_leaf: int = 2,
    ) -> None:
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        self.feature_names = tuple(FEATURES)
        self.class_labels = tuple(LABELS)
        self.is_trained = False
        self.metrics: Dict[str, object] | None = None
        self._test_rows: List[Dict[str, object]] = []
        self._X_test: List[List[object]] = []
        self._y_test: List[str] = []
        self._train_rows: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------
    def train(
        self,
        rows: Sequence[Dict[str, object]],
        test_size: float = 0.2,
    ) -> Dict[str, object]:
        """Split, fit on the training set and evaluate on the test set."""
        split = split_dataset(rows, test_size=test_size, random_state=self.random_state)
        X_train = split["X_train"]
        y_train = split["y_train"]
        self.model.fit(X_train, y_train)
        self.is_trained = True

        self._train_rows = split["train_rows"]  # type: ignore[assignment]
        self._test_rows = split["test_rows"]  # type: ignore[assignment]
        self._X_test = split["X_test"]  # type: ignore[assignment]
        self._y_test = split["y_test"]  # type: ignore[assignment]
        self.metrics = self._build_metrics(rows, split)
        return self.metrics

    def _build_metrics(self, rows: Sequence[Dict[str, object]], split: Dict[str, object]) -> Dict[str, object]:
        assert self.is_trained
        X_test = self._X_test
        y_test = self._y_test
        predictions = [str(label) for label in self.model.predict(X_test)]

        accuracy = float(accuracy_score(y_test, predictions))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
        matrix = confusion_matrix(y_test, predictions, labels=list(LABELS)).tolist()
        report = classification_report(
            y_test,
            predictions,
            labels=list(LABELS),
            target_names=list(LABELS),
            zero_division=0,
            output_dict=False,
        )

        false_predictions: List[Dict[str, object]] = []
        for row, truth, prediction in zip(self._test_rows, y_test, predictions):
            if truth != prediction:
                false_predictions.append(
                    {
                        "input_features": {name: row[name] for name in FEATURES},
                        "true_class": truth,
                        "predicted_class": prediction,
                        "possible_reason": _possible_reason(row, prediction),
                    }
                )

        all_labels = [str(row["label"]) for row in rows]
        return {
            "model": MODEL_NAME,
            "random_state": self.random_state,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "features": list(FEATURES),
            "labels": list(LABELS),
            "total_samples": len(rows),
            "training_samples": len(split["train_rows"]),  # type: ignore[arg-type]
            "testing_samples": len(split["test_rows"]),  # type: ignore[arg-type]
            "class_distribution": _class_counts(all_labels),
            "train_class_distribution": _class_counts(split["y_train"]),  # type: ignore[arg-type]
            "test_class_distribution": _class_counts(y_test),
            "accuracy": round(accuracy, 4),
            "precision_macro": round(float(precision), 4),
            "recall_macro": round(float(recall), 4),
            "f1_macro": round(float(f1), 4),
            "zero_division": 0,
            "confusion_matrix": matrix,
            "confusion_matrix_note": "Rows are true classes; columns are predicted classes, in NORMAL, SUSPICIOUS, CRITICAL order.",
            "classification_report": report,
            "false_prediction_count": len(false_predictions),
            "false_predictions": false_predictions,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_features(self, features: Sequence[object]) -> str:
        """Predict one five-feature record after validating it."""
        if not self.is_trained:
            raise RuntimeError("The classifier must be trained before prediction.")
        if not isinstance(features, (list, tuple)) or len(features) != len(FEATURES):
            raise ValueError(f"Expected {len(FEATURES)} features: {list(FEATURES)}")
        candidate = dict(zip(FEATURES, features))
        validation = validate_ml_features(candidate)
        if not validation["valid"]:
            raise ValueError("Invalid features: " + "; ".join(validation["errors"]))  # type: ignore[arg-type]
        prediction = self.model.predict([[candidate[name] for name in FEATURES]])[0]
        return str(prediction)

    def predict_event(self, event: Dict[str, object]) -> str:
        """Predict from an incident dictionary containing the five features."""
        validation = validate_ml_features(event)
        if not validation["valid"]:
            raise ValueError("Invalid ML features: " + "; ".join(validation["errors"]))  # type: ignore[arg-type]
        return self.predict_features([event[name] for name in FEATURES])

    def predict_with_confidence(self, event: Dict[str, object]) -> Dict[str, object]:
        """Return the predicted class plus the tree's class probabilities.

        Probabilities from a small synthetic decision tree are not calibrated
        confidence estimates; they are included only for demonstration.
        """
        prediction = self.predict_event(event)
        probabilities = self.model.predict_proba(
            [[event[name] for name in FEATURES]]
        )[0]
        classes = [str(label) for label in self.model.classes_]
        return {
            "predicted_class": prediction,
            "probabilities": {
                label: round(float(probabilities[classes.index(label)]), 4)
                for label in LABELS
                if label in classes
            },
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_metrics(self, path: Path | str) -> Path | None:
        if self.metrics is None:
            raise RuntimeError("No metrics available; train the model first.")
        output = Path(path)
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as handle:
                json.dump(self.metrics, handle, indent=2)
            return output
        except PermissionError as exc:
            print(f"WARNING: Could not save metrics to {output}: {exc}")
            print("         File may be read-only or locked. Metrics remain available in memory.")
            return None
        except OSError as exc:
            print(f"WARNING: Could not save metrics to {output}: {exc}")
            print("         Metrics remain available in memory.")
            return None


def train_default_model(
    dataset_path: Path | str | None = None,
    random_state: int = 42,
) -> IncidentClassifier:
    """Convenience function used by the CLI and tests."""
    from .dataset import DEFAULT_DATASET_PATH

    rows = load_dataset(dataset_path or DEFAULT_DATASET_PATH)
    model = IncidentClassifier(random_state=random_state)
    model.train(rows)
    return model
