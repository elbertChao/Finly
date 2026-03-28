import argparse
import os
import json
import time
from pathlib import Path

try:
    import openai
except ImportError:
    raise ImportError("openai package not installed. Install with pip install openai")

PROMPT_TEMPLATE = """
You are an expert, beginner-friendly financial analyst focusing on long-term value investing.
For the raw text below, extract core insights and format the output strictly into 4 sections:
1. Plain English Summary
2. Long-Term Bull Case
3. Long-Term Bear Case
4. Hold/Wait Analysis

Requirements:
- Keep tone accessible.
- Avoid heavy jargon.
- Focus on 5-10 year company horizon.
- If the source is unclear, say what data is missing.
- No bulletpoint floods; use concise paragraphs.

RAW TEXT:
{raw_text}
"""


def annotate_item(raw_text, model="gpt-4.1", max_retries=3):
    prompt = PROMPT_TEMPLATE.format(raw_text=raw_text)
    for attempt in range(1, max_retries + 1):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": "You are an expert financial analyst."},
                          {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1200,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            if attempt == max_retries:
                raise
            backoff = 2 ** attempt
            print(f"Retry {attempt}/{max_retries} after error: {exc}. Waiting {backoff}s")
            time.sleep(backoff)


def run(input_jsonl, output_jsonl, model):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY must be set in environment")
    openai.api_key = key

    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            item = json.loads(line)
            if item.get("response"):
                # skip already annotated
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            print(f"Annotating line {line_no}, id={item.get('metadata', {}).get('id', '')}")
            generated = annotate_item(item["context"], model=model)
            item["response"] = generated
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Annotation complete. Wrote output to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic gold-standard dataset output using OpenAI")
    parser.add_argument("--input-jsonl", default="data/curated_dataset.jsonl")
    parser.add_argument("--output-jsonl", default="data/curated_dataset_annotated.jsonl")
    parser.add_argument("--model", default="gpt-4.1", help="OpenAI model name")

    args = parser.parse_args()
    run(args.input_jsonl, args.output_jsonl, args.model)


if __name__ == "__main__":
    main()
