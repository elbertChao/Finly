# Finly

Finly is a domain-adapted large language model project focused on long-term value investing analysis. The goal is to fine-tune an instruction model on dense financial source material so it can convert raw filings, market coverage, and business commentary into structured, beginner-friendly investment insights.

The project is being built around an industry-style workflow:

- curate real-world financial text from reputable sources
- generate grounded teacher responses with a stronger LLM
- fine-tune a smaller open-weight model with LoRA or QLoRA
- later serve the model in a retrieval-augmented application for practical equity research

## Project Objective

Most retail investors can access financial information, but not necessarily interpret it well. Finly aims to narrow that gap by training a model to convert dense business and market text into a consistent analytical framework:

- Plain English Summary
- Long-Term Bull Case
- Long-Term Bear Case
- Hold/Wait Analysis

This structure is designed to make company research more accessible without flattening the nuance of the original source material.

## Current Pipeline

The repository currently covers the data and training-foundation stages of the workflow.

### 1. Dataset Curation

The dataset builder supports multiple ingestion paths so training data does not depend solely on manually prepared files.

Supported inputs:

- local `.txt` documents
- RSS feeds from financial or business news publishers
- article URLs listed in a text file
- SEC filings fetched directly from EDGAR

The curation script normalizes all raw inputs into a training-ready JSONL schema with:

- shared instruction text
- source context
- empty response field for later annotation
- metadata for source tracking and quality control

### 2. Gold-Standard Annotation

The annotation pipeline sends curated raw contexts into a stronger LLM, which generates structured target responses for supervised fine-tuning. This creates a teacher-student workflow while keeping inputs grounded in real financial source material.

### 3. LoRA or QLoRA Training

The training script is aligned to a Linux plus NVIDIA workflow and is intended to run on a CUDA-enabled machine such as an RTX 3060 environment.

## Repository Structure

```text
.
|-- scripts/
|   |-- curate_dataset.py
|   |-- generate_gold_standard.py
|   `-- train_qlora.py
|-- Fin-Instruct_details.txt
|-- Finly_details.txt
`-- README.md
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

### 5. Curate Dataset From SEC Filings

```bash
python scripts/curate_dataset.py \
  --sec-company "AAPL:320193" \
  --sec-company "MSFT:789019" \
  --sec-user-agent "Fin-Instruct research your-email@example.com" \
  --sec-form 10-K \
  --sec-form 10-Q \
  --sec-filings-per-company 2 \
  --output-jsonl data/curated_dataset.jsonl
```

This path pulls recent SEC filing documents directly from EDGAR and stores metadata such as ticker, CIK, form type, and filing date alongside each record.

### 6. Generate Gold-Standard Outputs

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

### 7. Train LoRA Adapter

```bash
python scripts/train_qlora.py \
  --dataset-path data/curated_dataset_annotated.jsonl \
  --output-dir artifacts/finly-lora
```

## Current Implementation Notes

- `scripts/curate_dataset.py` supports local text, RSS feeds, article URLs, and direct SEC filing ingestion.
- `scripts/generate_gold_standard.py` uses the current OpenAI client flow and supports retrying plus resumable output generation.
- `scripts/train_qlora.py` is aligned to a Linux plus NVIDIA training path instead of DirectML.
- The project is currently in the dataset and training-foundation stage, not yet deployment-ready.

## Quality Standards

The value of this project depends more on data quality than raw model size. Before training, curated examples should be reviewed for:

- grounding to the source text
- consistent section structure
- limited hallucination of figures or claims
- manageable context length for fine-tuning hardware limits

Manual spot-checking of randomly sampled records is a required part of the workflow.

## Roadmap

- expand source-specific collectors for high-value financial datasets
- introduce train and validation splitting plus lightweight evaluation
- package the resulting adapter for downstream inference
- add RAG-based retrieval with current market context
- build a deployable demo interface

## Environment Direction

The intended training environment is Linux with NVIDIA CUDA support. This repository is being prepared locally, while model fine-tuning is expected to run on a separate lab machine with an NVIDIA GPU.

## Security Note

Do not commit live API credentials into the repository. Store secrets in environment variables or local-only configuration and rotate any exposed keys immediately.
