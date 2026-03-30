import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Split annotated dataset JSONL into train and validation files")
    parser.add_argument("--input-jsonl", default="data/curated_dataset_annotated.jsonl")
    parser.add_argument("--train-output", default="data/train.jsonl")
    parser.add_argument("--validation-output", default="data/validation.jsonl")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not 0 < args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    records = load_jsonl(input_path)
    annotated_records = [record for record in records if str(record.get("response", "")).strip()]
    if len(annotated_records) < 2:
        raise ValueError("Need at least 2 annotated records to create train and validation splits.")

    rng = random.Random(args.seed)
    rng.shuffle(annotated_records)

    validation_size = max(1, int(len(annotated_records) * args.validation_ratio))
    validation_records = annotated_records[:validation_size]
    train_records = annotated_records[validation_size:]

    if not train_records:
        raise ValueError("Validation split consumed the full dataset. Lower --validation-ratio.")

    train_output = Path(args.train_output)
    validation_output = Path(args.validation_output)
    write_jsonl(train_output, train_records)
    write_jsonl(validation_output, validation_records)

    print(f"Input annotated records: {len(annotated_records)}")
    print(f"Train records: {len(train_records)} -> {train_output}")
    print(f"Validation records: {len(validation_records)} -> {validation_output}")


if __name__ == "__main__":
    main()
