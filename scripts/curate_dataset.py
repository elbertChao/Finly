import argparse
import json
from pathlib import Path


def scan_text_inputs(input_dir):
    data = []
    for path in sorted(Path(input_dir).rglob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            continue
        data.append({
            "id": path.stem,
            "source": str(path),
            "raw_text": raw,
        })
    return data


def build_jsonl_entries(samples, instruction):
    entries = []
    for s in samples:
        entries.append({
            "instruction": instruction,
            "context": s["raw_text"],
            "response": "",
            "metadata": {
                "id": s["id"],
                "source": s["source"],
            },
        })
    return entries


def write_jsonl(entries, output_path):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(entries)} examples to {p}")


def main():
    parser = argparse.ArgumentParser(description="Curate financial dataset JSONL")
    parser.add_argument("--input-dir", default="inputs", help="Directory containing raw text snippets (txt)")
    parser.add_argument("--output-jsonl", default="data/curated_dataset.jsonl", help="Output JSONL file path")
    parser.add_argument("--instruction", default="Analyze the long-term growth prospects of this company based on the text context.", help="Instruction to use for all examples")

    args = parser.parse_args()

    samples = scan_text_inputs(args.input_dir)
    if not samples:
        print("No text files found in input-dir. Drop text snippets into the folder and rerun.")
        return

    entries = build_jsonl_entries(samples, args.instruction)
    write_jsonl(entries, args.output_jsonl)


if __name__ == "__main__":
    main()
