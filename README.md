# Working time restricted search: if you want to build a forcaster



Generate search-augmented data with optional leakage validation for forecasting tasks.

## Overview

This tool augments datasets with relevant web articles fetched via Perplexity search. Optionally validates that articles don't contain information leakage (post-cutoff resolution information).

### Pipeline

1. **Perplexity Search** - Discover article URLs using Perplexity's Sonar/Pro Search
2. **Date Filtering** - Filter articles by publication date (before cutoff)
3. **Content Fetching** - Fetch full article content with trafilatura
4. **Leakage Validation** (optional) - Per-article LLM-as-judge leakage detection
5. **Relevance Scoring** - Score article relevance for the query
6. **Output** - JSONL with original data + validated articles

## Installation

```bash
uv add trafilatura perplexity-python safetytooling simple-parsing tqdm
```

## Environment Variables

```bash
export PERPLEXITY_API_KEY="pplx-..."  # Required for Perplexity search
export OPENAI_API_KEY="sk-..."        # Required for relevance scoring (and leakage checking if enabled)
```

## Usage

### Basic usage (no leakage checking)

```bash
uv run generate_search_augmented_data.py \
    --dataset_path data.jsonl \
    --query_field "question" \
    --date_field "created_at" \
    --output_dir output/ \
    --num_tasks 10
```

### With leakage checking

```bash
uv run generate_search_augmented_data.py \
    --dataset_path data.jsonl \
    --query_field "question" \
    --date_field "created_at" \
    --ground_truth_field "answer" \
    --output_dir output/ \
    --num_tasks 10
```

### Full example with all options

```bash
uv run generate_search_augmented_data.py \
    --dataset_path data.jsonl \
    --query_field "question" \
    --date_field "created_at" \
    --ground_truth_field "answer" \
    --output_dir output/ \
    --num_tasks 100 \
    --perplexity_model sonar-pro \
    --use_pro_search \
    --judge_model gpt-5-mini-2025-08-07 \
    --max_articles_per_market 10 \
    --cutoff_days_before_start 2 \
    --perplexity_concurrency 50 \
    --openai_num_threads 50 \
    --seed 42
```

## Input Format

JSONL file where each row contains at minimum:
- A query field (specified via `--query_field`)
- A date field (specified via `--date_field`)
- Optionally, a ground truth field for leakage checking (specified via `--ground_truth_field`)

Example:
```jsonl
{"question": "Will X happen?", "created_at": "2024-01-15", "answer": "Yes"}
{"question": "Will Y happen?", "created_at": "2024-02-01", "answer": "No"}
```

## Output Format

JSONL file with original data plus `search_context`:

```json
{
    "question": "...",
    "created_at": "...",
    "search_context": {
        "cutoff_date": "2024-01-13T00:00:00",
        "leakage_check_enabled": true,
        "num_articles_validated": 5,
        "num_articles_leaked": 1,
        "article_leakage_rate": 0.1667,
        "validated_articles": [
            {
                "url": "https://...",
                "title": "Article Title",
                "content": "Full article text...",
                "relevance_score": 85,
                "relevance_category": "high",
                "useful_info": ["Key fact 1", "Key fact 2"]
            }
        ]
    }
}
```

## Required Arguments

| Option | Description |
|--------|-------------|
| `--dataset_path` | Path to input JSONL dataset |
| `--query_field` | Column name containing the search query |
| `--date_field` | Column name containing the cutoff reference date |
| `--output_dir` | Output directory |

## Optional Arguments

| Option | Default | Description |
|--------|---------|-------------|
| `--ground_truth_field` | None | Column for ground truth (enables leakage checking) |
| `--num_tasks` | 10 | Number of rows to process |
| `--perplexity_model` | sonar-pro | Perplexity model (sonar, sonar-pro) |
| `--use_pro_search` | True | Use Pro Search (multi-step reasoning) |
| `--use_background_prompt` | False | Generate vague prompts via LLM |
| `--judge_model` | gpt-5-mini-2025-08-07 | Model for relevance/leakage judging |
| `--max_articles_per_market` | 10 | Max articles to fetch per row |
| `--cutoff_days_before_start` | 2 | Days before date field for cutoff |
| `--perplexity_concurrency` | 50 | Max concurrent Perplexity calls |
| `--article_fetch_concurrency` | 50 | Max concurrent article fetches |

## Features

- **Configurable fields** - Map your JSONL columns to query, date, and ground truth
- **Optional leakage checking** - Skip if you don't have ground truth labels
- **Field validation** - Errors on startup if specified fields don't exist
- **Resume support** - Automatically skips already-processed rows
- **Per-article validation** - LLM-as-judge validates each article individually
- **Relevance scoring** - Articles scored and categorized (high/medium/low/none)
- **Date filtering** - Articles after cutoff are excluded before fetching
- **Parallel processing** - Concurrent Perplexity searches and article fetches

## Leakage Reviewer

A simple web interface to manually review articles for information leakage.

```bash
uv run leakage_reviewer.py \
    --data_path search_augmented_data.jsonl \
    --query_field "question_title" \
    --answer_field "answer" \
    --num_samples 100 \
    --port 8765
```

Then open http://localhost:8765 in your browser.

Features:
- Shows one question at a time with top 3 articles
- Click articles to toggle between LEAKS/SAFE
- Annotations auto-save to `leakage_annotations.json`
- Progress tracking across sessions

## License

MIT
