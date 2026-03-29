import argparse
import json
import statistics
from pathlib import Path


REQUIRED_TOP_LEVEL_FIELDS = ["instruction", "context", "response", "metadata"]
REQUIRED_METADATA_FIELDS = ["id", "source"]
REQUIRED_RESPONSE_SECTIONS = [
    "Plain English Summary",
    "Long-Term Bull Case",
    "Long-Term Bear Case",
    "Hold/Wait Analysis",
]


def word_count(text: str) -> int:
    return len(text.split())


def summarize_lengths(values: list[int]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
    }


def validate_record(item: dict, require_response_sections: bool) -> list[str]:
    issues = []

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in item:
            issues.append(f"missing top-level field: {field}")

    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        issues.append("metadata must be an object")
        return issues

    for field in REQUIRED_METADATA_FIELDS:
        if not str(metadata.get(field, "")).strip():
            issues.append(f"missing metadata field: {field}")

    instruction = str(item.get("instruction", "")).strip()
    context = str(item.get("context", "")).strip()
    response = str(item.get("response", "")).strip()

    if not instruction:
        issues.append("instruction is empty")
    if not context:
        issues.append("context is empty")

    if require_response_sections and response:
        for section in REQUIRED_RESPONSE_SECTIONS:
            if section not in response:
                issues.append(f"response missing section heading: {section}")

    return issues


def run(
    dataset_path: str,
    require_annotated: bool,
    require_response_sections: bool,
    max_context_words: int | None,
    max_response_words: int | None,
    show_examples: int,
):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    total = 0
    empty_responses = 0
    invalid_records = []
    context_lengths = []
    response_lengths = []
    overlong_contexts = []
    overlong_responses = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            total += 1
            item = json.loads(line)
            issues = validate_record(item, require_response_sections=require_response_sections)

            context_words = word_count(str(item.get("context", "")).strip())
            response_words = word_count(str(item.get("response", "")).strip())
            context_lengths.append(context_words)
            response_lengths.append(response_words)

            if not str(item.get("response", "")).strip():
                empty_responses += 1

            if max_context_words is not None and context_words > max_context_words:
                overlong_contexts.append((line_no, context_words))
                issues.append(f"context exceeds max word count: {context_words} > {max_context_words}")

            if max_response_words is not None and response_words > max_response_words:
                overlong_responses.append((line_no, response_words))
                issues.append(f"response exceeds max word count: {response_words} > {max_response_words}")

            if require_annotated and not str(item.get("response", "")).strip():
                issues.append("response is empty but annotated dataset was required")

            if issues:
                invalid_records.append((line_no, issues))

    print(f"Dataset: {path}")
    print(f"Total records: {total}")
    print(f"Records with empty responses: {empty_responses}")
    print(f"Invalid records: {len(invalid_records)}")
    print(f"Context length summary (words): {summarize_lengths(context_lengths)}")
    print(f"Response length summary (words): {summarize_lengths(response_lengths)}")
    print(f"Contexts over limit: {len(overlong_contexts)}")
    print(f"Responses over limit: {len(overlong_responses)}")

    if invalid_records:
        print("\nSample issues:")
        for line_no, issues in invalid_records[:show_examples]:
            print(f"- line {line_no}: {'; '.join(issues)}")

    if invalid_records:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Validate curated or annotated dataset JSONL before annotation or training")
    parser.add_argument("--dataset-path", default="data/curated_dataset.jsonl")
    parser.add_argument("--require-annotated", action="store_true", help="Fail if any response field is empty")
    parser.add_argument(
        "--require-response-sections",
        action="store_true",
        help="Check annotated responses for the expected four section headings",
    )
    parser.add_argument("--max-context-words", type=int, default=1800)
    parser.add_argument("--max-response-words", type=int, default=700)
    parser.add_argument("--show-examples", type=int, default=10)

    args = parser.parse_args()
    run(
        dataset_path=args.dataset_path,
        require_annotated=args.require_annotated,
        require_response_sections=args.require_response_sections,
        max_context_words=args.max_context_words,
        max_response_words=args.max_response_words,
        show_examples=args.show_examples,
    )


if __name__ == "__main__":
    main()
