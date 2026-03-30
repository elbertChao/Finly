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

The training script is aligned to a Linux plus NVIDIA workflow and is intended to run in a CUDA-enabled environment.

## Repository Structure

```text
.
|-- scripts/
|   |-- curate_dataset.py
|   |-- evaluate_dataset.py
|   |-- generate_gold_standard.py
|   |-- split_dataset.py
|   |-- train_qlora.py
|   `-- validate_dataset.py
|-- .env.example
|-- Fin-Instruct_details.txt
|-- Finly_details.txt
|-- requirements.txt
`-- README.md
```

## Environment Setup

The recommended training target is a Linux environment with an NVIDIA GPU.

### 1. Create a Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install PyTorch for Your CUDA Version

Install PyTorch separately so it matches the CUDA version on your target system. Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If your system uses a different CUDA runtime, use the matching PyTorch install command from the official PyTorch site.

### 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Secrets

Copy `.env.example` to `.env` and provide your OpenAI key through the environment or local-only configuration.

On Linux:

```bash
export OPENAI_API_KEY="your_api_key_here"
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
  --sec-chunk-words 900 \
  --sec-max-chunks-per-filing 2 \
  --output-jsonl data/curated_dataset.jsonl
```

This path pulls recent SEC filing documents directly from EDGAR, extracts narrative sections such as MD&A, ranks the more financially relevant subsections, and stores chunk-level metadata such as ticker, CIK, form type, filing date, and chunk index alongside each record.

### 6. Generate Gold-Standard Outputs

Note: gpt-4.1-mini is used to save API Key costs to create a project, but in reality 4.1 or better should be used

```bash
python scripts/generate_gold_standard.py \
  --input-jsonl data/curated_dataset.jsonl \
  --output-jsonl data/curated_dataset_annotated.jsonl \
  --model gpt-4.1-mini
```

For smaller validation runs, annotate only part of the dataset first:

```bash
python scripts/generate_gold_standard.py \
  --input-jsonl data/curated_dataset.jsonl \
  --output-jsonl data/curated_dataset_annotated.jsonl \
  --model gpt-4.1-mini \
  --limit 25
```

### 7. Validate the Dataset Before Training

Validate a curated dataset before annotation:

```bash
python scripts/validate_dataset.py \
  --dataset-path data/curated_dataset.jsonl
```

Validate an annotated dataset before training:

```bash
python scripts/validate_dataset.py \
  --dataset-path data/curated_dataset_annotated.jsonl \
  --require-annotated \
  --require-response-sections
```

### 8. Split the Annotated Dataset

```bash
python scripts/split_dataset.py \
  --input-jsonl data/curated_dataset_annotated.jsonl \
  --train-output data/train.jsonl \
  --validation-output data/validation.jsonl \
  --validation-ratio 0.1
```

### 9. Run a Lightweight Dataset Evaluation

```bash
python scripts/evaluate_dataset.py \
  --dataset-path data/curated_dataset_annotated.jsonl \
  --examples 3
```

### 10. Train LoRA Adapter

```bash
python scripts/train_qlora.py \
  --train-dataset-path data/train.jsonl \
  --validation-dataset-path data/validation.jsonl \
  --output-dir artifacts/finly-lora
```

If GPU memory is tighter than expected, try shorter sequence lengths or disable 4-bit loading temporarily for debugging:

```bash
python scripts/train_qlora.py \
  --train-dataset-path data/train.jsonl \
  --validation-dataset-path data/validation.jsonl \
  --output-dir artifacts/finly-lora \
  --max-seq-length 768 \
  --no-4bit
```

## Current Implementation Notes

- `scripts/curate_dataset.py` supports local text, RSS feeds, article URLs, and direct SEC filing ingestion.
- `scripts/curate_dataset.py` can chunk SEC narrative sections into smaller ranked training examples instead of taking only one long filing slice.
- `scripts/generate_gold_standard.py` uses the current OpenAI client flow and supports retrying plus resumable output generation.
- `scripts/validate_dataset.py` checks dataset structure, empty responses, section headings, and rough length limits before annotation or training.
- `scripts/split_dataset.py` creates reproducible train and validation splits from an annotated dataset.
- `scripts/evaluate_dataset.py` provides lightweight reporting on dataset size, length distribution, source mix, and section coverage.
- `scripts/train_qlora.py` is aligned to a Linux plus NVIDIA training path and can evaluate against a validation split each epoch.
- `requirements.txt` captures the Python package dependencies, while PyTorch should be installed separately to match the CUDA environment on the target system.
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

The intended training environment is Linux with NVIDIA CUDA support.

## Security Note

Do not commit live API credentials into the repository. Store secrets in environment variables or local-only configuration and rotate any exposed keys immediately.
