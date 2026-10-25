from __future__ import annotations

import unittest

from sentinel.classifier import IncidentClassifier, split_dataset
from sentinel.dataset import DEFAULT_SEED, generate_records, load_dataset, validate_records
from sentinel.validation import validate_ml_features


class TestDataset(unittest.TestCase):
    def test_generates_at_least_300_records(self):
        rows = generate_records(400, seed=DEFAULT_SEED)
        self.assertGreaterEqual(len(rows), 300)

    def test_reproducible_generation(self):
        rows1 = generate_records(400, seed=42)
        rows2 = generate_records(400, seed=42)
        self.assertEqual(rows1, rows2)

    def test_valid_labels_and_all_classes(self):
        rows = generate_records(400, seed=42)
        validate_records(rows)
        labels = {row["label"] for row in rows}
        self.assertEqual(labels, {"NORMAL", "SUSPICIOUS", "CRITICAL"})

    def test_all_required_features_present(self):
        rows = generate_records(10, seed=42)
        for row in rows:
            for feature in ["failed_logins", "requests_per_min", "unusual_hour", "known_device", "duplicate_count", "label"]:
                self.assertIn(feature, row)

    def test_load_dataset_rejects_corruption(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset("data/does_not_exist.csv")


class TestMLClassifier(unittest.TestCase):
    def setUp(self):
        self.rows = generate_records(400, seed=42)

    def test_stratified_split(self):
        split = split_dataset(self.rows, test_size=0.2, random_state=42)
        self.assertEqual(len(split["X_train"]), len(split["y_train"]))
        self.assertEqual(len(split["X_test"]), len(split["y_test"]))
        self.assertEqual(len(split["X_train"]), 320)
        self.assertEqual(len(split["X_test"]), 80)
        # proportions
        train_counts = {label: split["y_train"].count(label) for label in ["NORMAL", "SUSPICIOUS", "CRITICAL"]}
        test_counts = {label: split["y_test"].count(label) for label in ["NORMAL", "SUSPICIOUS", "CRITICAL"]}
        self.assertGreater(train_counts["NORMAL"], 0)
        self.assertGreater(test_counts["NORMAL"], 0)

    def test_train_evaluate_no_label_leakage(self):
        model = IncidentClassifier(random_state=42)
        metrics = model.train(self.rows)
        # only 5 features
        self.assertEqual(len(model.feature_names), 5)
        self.assertNotIn("label", model.feature_names)
        self.assertTrue(model.is_trained)
        self.assertIn("accuracy", metrics)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(metrics["training_samples"], 320)
        self.assertEqual(metrics["testing_samples"], 80)

    def test_valid_predictions(self):
        model = IncidentClassifier(random_state=42)
        model.train(self.rows)
        event = {"failed_logins": 12, "requests_per_min": 120, "unusual_hour": 1, "known_device": 0, "duplicate_count": 4}
        validation = validate_ml_features(event)
        self.assertTrue(validation["valid"])
        prediction = model.predict_event(event)
        self.assertIn(prediction, ["NORMAL", "SUSPICIOUS", "CRITICAL"])

    def test_invalid_input_safely_handled(self):
        model = IncidentClassifier(random_state=42)
        model.train(self.rows)
        with self.assertRaises(ValueError):
            model.predict_event({"failed_logins": -1, "requests_per_min": 10, "unusual_hour": 0, "known_device": 1, "duplicate_count": 0})


if __name__ == "__main__":
    unittest.main()
