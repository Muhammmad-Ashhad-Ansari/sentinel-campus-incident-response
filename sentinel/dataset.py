"""Deterministic synthetic incident dataset for SENTINEL.

The data is synthetic, not real campus security data.  A fixed seed makes the
CSV reproducible.  The 12 starter rows from the assignment are kept at the top
of the file; the remaining rows are generated from the documented rules below.

Generation rules (probabilities are per record):

NORMAL (45%)
    failed_logins 0..3, requests_per_min 5..35,
    unusual_hour 1 with 8%, known_device 1 with 92%, duplicate_count 0..1

SUSPICIOUS (35%)
    failed_logins 2..9, requests_per_min 25..90,
    unusual_hour 1 with 55%, known_device 1 with 45%, duplicate_count 0..3

CRITICAL (20%)
    failed_logins 7..20, requests_per_min 60..160,
    unusual_hour 1 with 85%, known_device 1 with 12%, duplicate_count 2..8

The ranges deliberately overlap.  That overlap creates realistic boundary
cases so the classifier is not a perfect lookup table.

Limitations: the rules are invented for a teaching exercise.  They do not
model real attackers, real campus traffic, or real base rates.  High accuracy
on this CSV therefore proves only that the pipeline works.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Sequence

FEATURES: Sequence[str] = (
    "failed_logins",
    "requests_per_min",
    "unusual_hour",
    "known_device",
    "duplicate_count",
)
LABELS: Sequence[str] = ("NORMAL", "SUSPICIOUS", "CRITICAL")
CSV_FIELDS: Sequence[str] = FEATURES + ("label",)
DEFAULT_SEED = 42
DEFAULT_RECORD_COUNT = 400

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "incidents.csv"

# The exact 12-row starter dataset from the assignment.  It is useful for a
# quick pipeline check, but far too small to train a credible model on its own.
STARTER_ROWS = [
    [0, 5, 0, 1, 0, "NORMAL"],
    [1, 8, 0, 1, 0, "NORMAL"],
    [0, 12, 0, 1, 1, "NORMAL"],
    [3, 25, 1, 1, 0, "SUSPICIOUS"],
    [5, 40, 1, 0, 1, "SUSPICIOUS"],
    [2, 32, 0, 0, 2, "SUSPICIOUS"],
    [8, 90, 1, 0, 3, "CRITICAL"],
    [12, 120, 1, 0, 4, "CRITICAL"],
    [10, 100, 0, 0, 5, "CRITICAL"],
    [4, 15, 0, 1, 0, "NORMAL"],
    [6, 55, 1, 0, 2, "SUSPICIOUS"],
    [15, 150, 1, 0, 8, "CRITICAL"],
]


def _sample_features(rng: random.Random, label: str) -> Dict[str, int]:
    """Draw one record from the documented class-conditional ranges."""
    if label == "NORMAL":
        return {
            "failed_logins": rng.randint(0, 3),
            "requests_per_min": rng.randint(5, 35),
            "unusual_hour": 1 if rng.random() < 0.08 else 0,
            "known_device": 1 if rng.random() < 0.92 else 0,
            "duplicate_count": rng.randint(0, 1),
        }
    if label == "SUSPICIOUS":
        return {
            "failed_logins": rng.randint(2, 9),
            "requests_per_min": rng.randint(25, 90),
            "unusual_hour": 1 if rng.random() < 0.55 else 0,
            "known_device": 1 if rng.random() < 0.45 else 0,
            "duplicate_count": rng.randint(0, 3),
        }
    if label == "CRITICAL":
        return {
            "failed_logins": rng.randint(7, 20),
            "requests_per_min": rng.randint(60, 160),
            "unusual_hour": 1 if rng.random() < 0.85 else 0,
            "known_device": 1 if rng.random() < 0.12 else 0,
            "duplicate_count": rng.randint(2, 8),
        }
    raise ValueError(f"Unknown label: {label!r}")


def _class_targets(total: int) -> Dict[str, int]:
    """Split the requested count into the documented 45/35/20 class mix."""
    normal = round(total * 0.45)
    suspicious = round(total * 0.35)
    critical = total - normal - suspicious
    return {"NORMAL": normal, "SUSPICIOUS": suspicious, "CRITICAL": critical}


def generate_records(
    count: int = DEFAULT_RECORD_COUNT,
    seed: int = DEFAULT_SEED,
    include_starter: bool = True,
) -> List[Dict[str, object]]:
    """Generate a deterministic list of labelled incident records.

    ``count`` includes the 12 starter rows when ``include_starter`` is true and
    ``count >= 12``.  The same arguments always produce the same records.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")

    rows: List[Dict[str, object]] = []
    if include_starter and count >= len(STARTER_ROWS):
        for values in STARTER_ROWS:
            rows.append(dict(zip(CSV_FIELDS, values)))

    targets = _class_targets(count)
    already = {label: 0 for label in LABELS}
    for row in rows:
        already[str(row["label"])] += 1

    rng = random.Random(seed)
    # Generate class blocks in a fixed order for reproducibility.  Shuffling
    # afterwards is seeded too, so the file order is stable across runs.
    for label in LABELS:
        needed = max(0, targets[label] - already[label])
        for _ in range(needed):
            record = _sample_features(rng, label)
            record["label"] = label
            rows.append(record)

    # If the caller asked for a count smaller than the starter set, trim it.
    rows = rows[:count]
    # Fill any shortfall caused by rounding or trimming with the largest class.
    if len(rows) < count:
        label = "SUSPICIOUS"
        while len(rows) < count:
            record = _sample_features(rng, label)
            record["label"] = label
            rows.append(record)

    rng.shuffle(rows)
    return rows


def validate_records(rows: Sequence[Dict[str, object]]) -> None:
    """Raise ValueError if a generated/loaded record is malformed."""
    if not rows:
        raise ValueError("Dataset is empty.")
    for index, row in enumerate(rows, start=1):
        missing = [field for field in CSV_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Row {index} is missing fields: {missing}")
        label = row["label"]
        if label not in LABELS:
            raise ValueError(f"Row {index} has invalid label: {label!r}")
        if not isinstance(row["failed_logins"], int) or row["failed_logins"] < 0:
            raise ValueError(f"Row {index} has invalid failed_logins.")
        if not isinstance(row["requests_per_min"], (int, float)) or row["requests_per_min"] < 0:
            raise ValueError(f"Row {index} has invalid requests_per_min.")
        for field in ("unusual_hour", "known_device"):
            if row[field] not in (0, 1):
                raise ValueError(f"Row {index} has invalid {field}.")
        if not isinstance(row["duplicate_count"], int) or row["duplicate_count"] < 0:
            raise ValueError(f"Row {index} has invalid duplicate_count.")


def write_dataset(
    rows: Sequence[Dict[str, object]],
    path: Path | str = DEFAULT_DATASET_PATH,
) -> Path:
    """Write the CSV, replacing any previous file (never appending duplicates)."""
    validate_records(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temporary.replace(output)  # atomic replacement on the same filesystem
    return output


def load_dataset(path: Path | str = DEFAULT_DATASET_PATH) -> List[Dict[str, object]]:
    """Read and validate the CSV, converting numeric columns to numbers."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    rows: List[Dict[str, object]] = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != set(CSV_FIELDS):
            raise ValueError(f"Dataset columns must be exactly: {list(CSV_FIELDS)}")
        for raw in reader:
            row: Dict[str, object] = {
                "failed_logins": int(raw["failed_logins"]),
                "requests_per_min": float(raw["requests_per_min"]),
                "unusual_hour": int(raw["unusual_hour"]),
                "known_device": int(raw["known_device"]),
                "duplicate_count": int(raw["duplicate_count"]),
                "label": raw["label"],
            }
            rows.append(row)
    validate_records(rows)
    return rows


def ensure_dataset(
    path: Path | str = DEFAULT_DATASET_PATH,
    count: int = DEFAULT_RECORD_COUNT,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> List[Dict[str, object]]:
    """Generate and save the dataset, or load an existing valid CSV."""
    output = Path(path)
    if output.exists() and not force:
        try:
            rows = load_dataset(output)
            if len(rows) >= 300:
                return rows
        except (ValueError, OSError):
            pass  # regenerate a corrupt or outdated file
    rows = generate_records(count=count, seed=seed)
    write_dataset(rows, output)
    return rows
