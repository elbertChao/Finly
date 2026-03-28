# Fin-Instruct

An LLM fine-tuned specifically to ingest dense financial texts to output highly structured, beginner-friendly insights focusing on long-term growth and hold strategies for stocks & investments.

## Quickstart (Dataset + Synthetic Gold Standard)

1. Create raw text snippets:
   - Place `.txt` files into `inputs/` (e.g. `inputs/earnings_q4_2025.txt`).
   - Each file should have a single raw financial text sample (10-K/10-Q MD&A, earnings call excerpt, etc.).

2. Build curated dataset JSONL:
   - `python scripts/curate_dataset.py --input-dir inputs --output-jsonl data/curated_dataset.jsonl`

3. Generate synthetic teacher outputs (OpenAI):
   - Ensure `OPENAI_API_KEY` is set in your environment.
   - `python scripts/generate_gold_standard.py --input-jsonl data/curated_dataset.jsonl --output-jsonl data/curated_dataset_annotated.jsonl --model gpt-4.1`

4. Validate samples:
   - Check `data/curated_dataset_annotated.jsonl` for 4 required sections in `response`.
   - Spot-check 50+ examples for grounding/hallucination.

5. Next steps:
   - Train QLoRA with `trl`/`bitsandbytes` using annotated dataset.
   - Build RAG pipeline with LangChain + finance APIs.
   - Deploy with FastAPI/Streamlit and Hugging Face Spaces.
