# Finly

Finly is a domain-adapted large language model project focused on long-term value investing analysis. The goal is to fine-tune an instruction model on dense financial source material so it can turn raw filings, market coverage, and business commentary into structured, beginner-friendly investment insights.

The project is designed around an industry-style workflow:

- curate real-world financial text from reputable sources
- generate grounded "teacher" responses with a stronger LLM
- fine-tune a smaller open-weight model with LoRA/QLoRA
- later serve the model in a retrieval-augmented application for practical equity research

## Project Objective

Most retail investors can access financial information, but not necessarily interpret it. Fin-Instruct aims to reduce that gap by training a model to convert dense business and market text into a consistent framework:

- Plain English Summary
- Long-Term Bull Case
- Long-Term Bear Case
- Hold/Wait Analysis

This structure is intended to make company research more accessible without flattening the nuance of the original source material.

## Current Pipeline

The repository currently covers the first two stages of the training workflow.

### 1. Dataset Curation

The dataset builder supports multiple ingestion paths so training data does not depend solely on manually prepared files.

Supported inputs:

- local `.txt` documents
- RSS feeds from financial/news publishers
- article URLs listed in a text file

The curation script normalizes all raw inputs into a training-ready JSONL schema with:

- shared instruction text
- source context
- empty response field for later annotation
- metadata for source tracking and quality control

### 2. Gold-Standard Annotation

The annotation pipeline is designed to pass curated raw contexts into a stronger LLM, which generates structured target responses for supervised fine-tuning. This creates a synthetic teacher-student workflow while still grounding the inputs in real source material.

### 3. LoRA / QLoRA Training

The training script has been refactored for a Linux + NVIDIA workflow and is intended to run on a CUDA-enabled machine such as an RTX 3060 environment.

## Repository Structure

```text
.
├── scripts/
│   ├── curate_dataset.py
│   ├── generate_gold_standard.py
│   └── train_qlora.py
└── README.md
```

## Data Schema

The curation step produces JSONL rows in the following format:

```json
{
  "instruction": "Analyze the long-term growth prospects of this company based on the text context.",
  "context": "Raw financial source text...",
  "response": "",
  "metadata": {
    "id": "sample-id",
    "source": "https://example.com/article",
    "title": "Article title",
    "source_type": "article_url"
  }
}
```

After annotation, the `response` field contains the teacher model output used for supervised fine-tuning.

## Usage

### 1. Curate Dataset From Local Text

```bash
python scripts/curate_dataset.py --input-dir inputs --output-jsonl data/curated_dataset.jsonl
```

### 2. Curate Dataset From RSS Feeds

```bash
python scripts/curate_dataset.py \
  --feed-url "https://example.com/feed.xml" \
  --feed-item-limit 20 \
  --output-jsonl data/curated_dataset.jsonl
```

### 3. Curate Dataset From Article URLs

Create a file such as `config/article_urls.txt` with one URL per line, then run:

```bash
python scripts/curate_dataset.py \
  --url-file config/article_urls.txt \
  --output-jsonl data/curated_dataset.jsonl
```

### 4. Mix Multiple Source Types

```bash
python scripts/curate_dataset.py \
  --input-dir inputs \
  --feed-url "https://example.com/feed.xml" \
  --url-file config/article_urls.txt \
  --output-jsonl data/curated_dataset.jsonl
```

### 5. Generate Gold-Standard Outputs

```bash
python scripts/generate_gold_standard.py \
  --input-jsonl data/curated_dataset.jsonl \
  --output-jsonl data/curated_dataset_annotated.jsonl \
  --model gpt-4.1
```

For smaller validation runs, annotate only part of the dataset first:

```bash
python scripts/generate_gold_standard.py \
  --input-jsonl data/curated_dataset.jsonl \
  --output-jsonl data/curated_dataset_annotated.jsonl \
  --model gpt-4.1 \
  --limit 25
```

### 6. Train LoRA Adapter

```bash
python scripts/train_qlora.py \
  --dataset-path data/curated_dataset_annotated.jsonl \
  --output-dir artifacts/finly-lora
```

## Current Implementation Notes

- `scripts/curate_dataset.py` supports online ingestion and local fallback input.
- `scripts/generate_gold_standard.py` uses the current OpenAI client flow and supports retrying plus resumable output generation.
- `scripts/train_qlora.py` is aligned to a Linux + NVIDIA training path instead of DirectML.
- The project is currently in the dataset and training-foundation stage, not yet deployment-ready.

## Quality Standards

The value of this project depends more on data quality than raw model size. Before training, curated examples should be reviewed for:

- grounding to the source text
- consistent section structure
- limited hallucination of figures or claims
- manageable context length for fine-tuning hardware limits

Manual spot-checking of randomly sampled records is a required part of the workflow.

## Roadmap

- modernize the gold-standard generation script and improve retry/resume behavior
- add source-specific collectors for high-value financial datasets
- introduce train/validation splitting and lightweight evaluation
- package the resulting adapter for downstream inference
- add RAG-based retrieval with current market context
- build a deployable demo interface

## Environment Direction

The intended training environment is Linux with NVIDIA CUDA support.

## Security Note

For other contributors or anyone that wants to use this repo, do not commit live API credentials into the repository. Store secrets in environment variables or local-only configuration and rotate any exposed keys immediately.
