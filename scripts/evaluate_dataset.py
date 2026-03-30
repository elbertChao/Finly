import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path


REQUIRED_RESPONSE_SECTIONS = [
    "Plain English Summary",
    "Long-Term Bull Case",
    "Long-Term Bear Case",
    "Hold/Wait Analysis",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def word_count(text: str) -> int:
    return len(str(text).split())


def summarize(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
    }


def section_coverage(records: list[dict]) -> dict[str, int]:
    coverage = {section: 0 for section in REQUIRED_RESPONSE_SECTIONS}
    for record in records:
        response = str(record.get("response", ""))
        for section in REQUIRED_RESPONSE_SECTIONS:
            if section in response:
                coverage[section] += 1
    return coverage


def print_examples(records: list[dict], count: int, seed: int):
    if not records or count <= 0:
        return

    rng = random.Random(seed)
    sample = records[:] if len(records) <= count else rng.sample(records, count)
    print("\nSample records:")
    for index, record in enumerate(sample, start=1):
        metadata = record.get("metadata") or {}
        source_type = metadata.get("source_type", "unknown")
        title = metadata.get("title", "") or "untitled"
        print(f"- sample {index}: title={title} | source_type={source_type} | id={metadata.get('id', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="Lightweight evaluation report for curated or annotated dataset JSONL")
    parser.add_argument("--dataset-path", default="data/curated_dataset_annotated.jsonl")
    parser.add_argument("--examples", type=int, default=3, help="Number of random sample records to print")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    records = load_jsonl(dataset_path)
    if not records:
        raise ValueError("Dataset is empty.")

    context_lengths = [word_count(record.get("context", "")) for record in records]
    response_lengths = [word_count(record.get("response", "")) for record in records]
    source_types = Counter((record.get("metadata") or {}).get("source_type", "unknown") for record in records)
    annotated_records = [record for record in records if str(record.get("response", "")).strip()]
    coverage = section_coverage(annotated_records)

    print(f"Dataset: {dataset_path}")
    print(f"Total records: {len(records)}")
    print(f"Annotated records: {len(annotated_records)}")
    print(f"Unannotated records: {len(records) - len(annotated_records)}")
    print(f"Context length summary (words): {summarize(context_lengths)}")
    print(f"Response length summary (words): {summarize(response_lengths)}")
    print(f"Source type distribution: {dict(source_types)}")
    print(f"Response section coverage: {coverage}")

    print_examples(records, count=args.examples, seed=args.seed)


if __name__ == "__main__":
    main()
