import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv


SYSTEM_PROMPT = """
You are an expert financial analyst and dataset annotator specializing in long-term value investing.
Your job is to transform raw financial source text into grounded, beginner-friendly instructional outputs.

You must follow these rules:
- Use only information supported by the provided source text.
- Do not invent figures, dates, management claims, or market facts.
- If the source is incomplete or ambiguous, explicitly say what is missing.
- Keep the tone accessible to non-expert investors.
- Focus on long-term business quality and 5-10 year investing implications.
- Avoid excessive bullet lists and keep each section concise but informative.
- Return plain text only.
"""


USER_PROMPT_TEMPLATE = """
Read the source text and produce exactly these four sections with these headings:

1. Plain English Summary
2. Long-Term Bull Case
3. Long-Term Bear Case
4. Hold/Wait Analysis

Write a grounded, structured response based only on the provided source.

Optional source title: {title}
Optional source type: {source_type}
Optional source location: {source}

SOURCE TEXT:
{raw_text}
"""


def load_client() -> OpenAI:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in the environment.")
    return OpenAI(api_key=api_key)


def annotate_item(
    client: OpenAI,
    raw_text: str,
    model: str,
    title: str,
    source_type: str,
    source: str,
    max_output_tokens: int,
    max_retries: int,
) -> str:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        raw_text=raw_text,
        title=title or "N/A",
        source_type=source_type or "N/A",
        source=source or "N/A",
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_output_tokens=max_output_tokens,
            )
            return response.output_text.strip()
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Annotation failed after {max_retries} attempts: {exc}") from exc

            backoff_seconds = min(2 ** attempt, 30)
            print(f"Retry {attempt}/{max_retries} after error: {exc}. Waiting {backoff_seconds}s")
            time.sleep(backoff_seconds)


def load_existing_output(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}

    existing = {}
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            metadata = item.get("metadata") or {}
            item_id = metadata.get("id")
            if item_id:
                existing[item_id] = item
    return existing


def ensure_metadata_id(item: dict, line_no: int) -> str:
    metadata = item.setdefault("metadata", {})
    item_id = metadata.get("id")
    if not item_id:
        item_id = f"line-{line_no}"
        metadata["id"] = item_id
    return item_id


def write_jsonl(path: Path, items: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def run(
    input_jsonl: str,
    output_jsonl: str,
    model: str,
    max_output_tokens: int,
    max_retries: int,
    limit: int | None,
):
    client = load_client()
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    existing_by_id = load_existing_output(output_path)
    final_items = []
    annotated_count = 0
    reused_count = 0

    with input_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            item_id = ensure_metadata_id(item, line_no)

            if limit is not None and len(final_items) >= limit:
                break

            existing = existing_by_id.get(item_id)
            if existing and (existing.get("response") or "").strip():
                final_items.append(existing)
                reused_count += 1
                continue

            response_text = (item.get("response") or "").strip()
            if response_text:
                final_items.append(item)
                reused_count += 1
                continue

            metadata = item.get("metadata") or {}
            print(f"Annotating line {line_no}, id={item_id}")
            item["response"] = annotate_item(
                client=client,
                raw_text=item["context"],
                model=model,
                title=metadata.get("title", ""),
                source_type=metadata.get("source_type", ""),
                source=metadata.get("source", ""),
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
            )
            final_items.append(item)
            annotated_count += 1

            write_jsonl(output_path, final_items)

    write_jsonl(output_path, final_items)
    print(
        f"Annotation complete. Wrote {len(final_items)} records to {output_path} "
        f"({annotated_count} newly annotated, {reused_count} reused)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate grounded gold-standard responses for a curated financial dataset using OpenAI"
    )
    parser.add_argument("--input-jsonl", default="data/curated_dataset.jsonl")
    parser.add_argument("--output-jsonl", default="data/curated_dataset_annotated.jsonl")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model name")
    parser.add_argument("--max-output-tokens", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Annotate only the first N records")

    args = parser.parse_args()
    run(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        max_retries=args.max_retries,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
