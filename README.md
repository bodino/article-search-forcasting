# Article Search for Forecasting

Generate search-augmented market data with leakage validation for prediction market forecasting.

## Overview

This tool augments prediction market datasets with relevant web articles while ensuring no information leakage (i.e., no articles containing post-cutoff resolution information).

### Pipeline

1. **Perplexity Search** - Discover article URLs using Perplexity's Sonar/Pro Search
2. **Date Filtering** - Filter articles by publication date (before market cutoff)
3. **Content Fetching** - Fetch full article content with trafilatura
4. **Leakage Validation** - Per-article LLM-as-judge leakage detection
5. **Relevance Scoring** - Score article relevance for forecasting
6. **Output** - JSONL with original market data + validated articles

## Installation

```bash
# Using uv (recommended)
uv add trafilatura perplexity-python safetytooling simple-parsing tqdm

# Or with pip
pip install trafilatura perplexity-python safetytooling simple-parsing tqdm
```

## Environment Variables

```bash
export PERPLEXITY_API_KEY="pplx-..."  # Required for Perplexity search
export OPENAI_API_KEY="sk-..."        # Required for leakage/relevance judging
```

## Usage

### Basic usage

```bash
uv run generate_search_augmented_data.py \
    --dataset_path market_data/test.jsonl \
    --output_dir data/experiments/search_augmented \
    --num_tasks 10 \
    --openai_num_threads 20
```

### Full example with all options

```bash
uv run generate_search_augmented_data.py \
    --dataset_path market_data/splits/test.jsonl \
    --output_dir data/experiments/search_augmented \
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

### Using background prompts (avoids outcome-seeking queries)

```bash
uv run generate_search_augmented_data.py \
    --dataset_path market_data/test.jsonl \
    --output_dir data/experiments/search_augmented \
    --use_background_prompt \
    --prompt_model gpt-5-mini-2025-08-07 \
    --num_tasks 10
```

## Input Format

JSONL file with prediction market data. Each line must contain:

```json
{
    "question_title": "Will X happen by Y date?",
    "question_start_date": "2024-01-15",
    "answer": "Yes",
    "resolution_criteria": "Resolves YES if..."
}
```

## Output Format

JSONL file with original market data plus `search_context`:

```json
{
    "question_title": "...",
    "answer": "...",
    "search_context": {
        "cutoff_date": "2024-01-13T00:00:00",
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

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset_path` | Required | Path to input JSONL dataset |
| `--output_dir` | Required | Output directory |
| `--num_tasks` | 10 | Number of markets to process |
| `--perplexity_model` | sonar-pro | Perplexity model (sonar, sonar-pro) |
| `--use_pro_search` | True | Use Pro Search (multi-step reasoning) |
| `--use_background_prompt` | False | Generate vague prompts via LLM |
| `--judge_model` | gpt-5-mini-2025-08-07 | Model for leakage/relevance judging |
| `--max_articles_per_market` | 10 | Max articles to fetch per market |
| `--cutoff_days_before_start` | 2 | Days before market start for cutoff |
| `--perplexity_concurrency` | 50 | Max concurrent Perplexity calls |
| `--article_fetch_concurrency` | 50 | Max concurrent article fetches |

## Features

- **Resume support** - Automatically skips already-processed markets
- **Per-article leakage detection** - LLM-as-judge validates each article individually
- **Relevance scoring** - Articles scored and categorized (high/medium/low/none)
- **Date filtering** - Articles after cutoff are excluded before fetching
- **Parallel processing** - Concurrent Perplexity searches and article fetches

## License

MIT
