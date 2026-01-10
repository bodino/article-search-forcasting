#!/usr/bin/env python
"""
Generate search-augmented market data with leakage validation.

Pipeline:
1. Perform Perplexity search to discover article URLs
2. Filter articles by publication date (before cutoff)
3. Fetch full article content with trafilatura
4. Validate no information leakage per-article via LLM-as-judge
5. Output JSONL with original data + validated articles

Example:
    uv run generate_search_augmented_data.py \
        --dataset_path market_data/splits/test.jsonl \
        --output_dir data/experiments/search_augmented \
        --num_tasks 10 \
        --openai_num_threads 20
"""

import asyncio
import json
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import trafilatura
from perplexity import Perplexity
from safetytooling.apis import InferenceAPI
from safetytooling.utils.experiment_utils import ExperimentConfigBase
from simple_parsing import ArgumentParser
from tqdm.asyncio import tqdm_asyncio

LOGGER = logging.getLogger(__name__)


# Wrapper function for safetytooling API calls
async def ask_single_question(
    api: InferenceAPI,
    model_id: str,
    question: str,
    system_prompt: str | None = None,
    **api_kwargs: Any,
) -> list[str]:
    """Wrapper around InferenceAPI.ask_single_question that handles provider prefixes.

    If model_id contains a colon, everything before the colon is treated as the provider
    and force_provider=True is set. For example:
    - "openai:gpt-4o-2024-08-06" -> force_provider="openai", model_id="gpt-4o-2024-08-06"
    - "anthropic:claude-3-5-sonnet-20240620" -> force_provider="anthropic", model_id="claude-3-5-sonnet-20240620"
    """
    if ":" in model_id:
        provider, actual_model_id = model_id.split(":", 1)
        api_kwargs["force_provider"] = provider
        model_id = actual_model_id

    return await api.ask_single_question(
        model_id=model_id,
        question=question,
        system_prompt=system_prompt,
        **api_kwargs,
    )


@dataclass(kw_only=True)
class SearchAugmentConfig(ExperimentConfigBase):
    """Configuration for search-augmented data generation."""

    dataset_path: str = field(metadata={"help": "Path to input JSONL dataset"})
    num_tasks: int = field(default=10, metadata={"help": "Number of markets to process"})
    num_searches_per_market: int = field(
        default=1, metadata={"help": "Number of Perplexity searches per market"}
    )
    perplexity_concurrency: int = field(
        default=50, metadata={"help": "Max concurrent Perplexity API calls"}
    )
    article_fetch_concurrency: int = field(
        default=50, metadata={"help": "Max concurrent article fetches with trafilatura (uses processes)"}
    )
    cutoff_days_before_start: int = field(
        default=2, metadata={"help": "Days before market start for search cutoff"}
    )
    perplexity_model: str = field(default="sonar-pro", metadata={"help": "Perplexity model to use (sonar, sonar-pro)"})
    use_pro_search: bool = field(default=True, metadata={"help": "Use Pro Search (requires sonar-pro model)"})
    use_background_prompt: bool = field(default=False, metadata={"help": "Generate vague background prompts via LLM instead of direct questions"})
    prompt_model: str = field(default="gpt-5-mini-2025-08-07", metadata={"help": "Model for generating background prompts"})
    judge_model: str = field(
        default="gpt-5-mini-2025-08-07", metadata={"help": "Model for per-article leakage judgment"}
    )
    max_articles_per_market: int = field(
        default=10, metadata={"help": "Maximum articles to fetch per market"}
    )
    max_article_length: int = field(
        default=10000, metadata={"help": "Maximum article content length (chars) to send to judge"}
    )


@dataclass
class SearchResult:
    """Result from Perplexity search with validation."""

    query: str
    response_text: str
    cutoff_date: str
    citations: list[str]
    leakage_check: dict
    is_valid: bool  # True if no leakage detected


@dataclass
class PerplexityProSearchResult:
    """Result from Perplexity Pro Search."""

    response_text: str
    citations: list[str]
    search_results: list[dict]  # Articles with title, url, date, snippet
    reasoning_steps: list[dict]  # Multi-step reasoning from Pro Search
    fetched_contents: list[dict]  # Content fetched via fetch_url_content tool


def _sync_perplexity_search(
    query: str,
    before_date: datetime | None = None,
    model: str = "sonar-pro",
    use_pro_search: bool = True,
) -> PerplexityProSearchResult:
    """
    Synchronous Perplexity search using SDK (called via asyncio.to_thread).

    Pro Search (use_pro_search=True) provides:
    - Multi-step reasoning with automated tools
    - search_results: List of articles with title, url, date, snippet
    - reasoning_steps: Tool usage and thought process (including fetch_url_content)

    Basic Sonar (use_pro_search=False) provides:
    - Simple single-step search
    - Citations only (no detailed search_results or reasoning)

    Args:
        query: The search query
        before_date: Only include results last modified before this date
        model: Perplexity model to use
        use_pro_search: Whether to use Pro Search features
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable required")

    if not api_key.startswith("pplx-"):
        raise ValueError(
            f"Invalid PERPLEXITY_API_KEY format (starts with '{api_key[:5]}...'). "
            "Valid keys start with 'pplx-'"
        )

    client = Perplexity(api_key=api_key)

    messages = [{"role": "user", "content": query}]

    # Build API call kwargs - date filters are TOP-LEVEL parameters
    api_kwargs = {
        "model": model,
        "messages": messages,
        "search_domain_filter": ["-wikipedia.org"],
    }

    # Pro Search requires streaming and specific web_search_options
    if use_pro_search:
        api_kwargs["stream"] = True
        api_kwargs["web_search_options"] = {"search_type": "pro"}
        search_type_str = "Pro Search"
    else:
        # Basic Sonar - no streaming needed, simpler request
        api_kwargs["stream"] = True  # Still use streaming for consistency
        search_type_str = "Basic Sonar"

    # Date filters are top-level parameters (format: M/D/YYYY without leading zeros)
    if before_date:
        date_str = f"{before_date.month}/{before_date.day}/{before_date.year}"
        api_kwargs["search_before_date_filter"] = date_str

    # Log the query and cutoff for debugging
    print(f"[Perplexity {search_type_str}] Model: {model}")
    print(f"[Perplexity {search_type_str}] Query: {query[:100]}...")
    if before_date:
        print(f"[Perplexity {search_type_str}] search_before_date_filter: {date_str}")
    else:
        print(f"[Perplexity {search_type_str}] No date filter")

    # Pro Search requires streaming
    response_stream = client.chat.completions.create(**api_kwargs)

    # Collect streamed response
    response_text_parts = []
    search_results = []
    reasoning_steps = []
    citations = []
    final_chunk = None

    for chunk in response_stream:
        final_chunk = chunk

        # Collect text content
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                response_text_parts.append(delta.content)

        # Extract search_results from chunk (available in final chunks)
        if hasattr(chunk, "search_results") and chunk.search_results:
            search_results = chunk.search_results

        # Extract reasoning_steps from chunk
        if hasattr(chunk, "reasoning_steps") and chunk.reasoning_steps:
            reasoning_steps = chunk.reasoning_steps

        # Extract citations
        if hasattr(chunk, "citations") and chunk.citations:
            citations = chunk.citations

    # Try to get search_results and reasoning_steps from final chunk if not collected
    if final_chunk:
        if not search_results and hasattr(final_chunk, "search_results"):
            search_results = final_chunk.search_results or []
        if not reasoning_steps and hasattr(final_chunk, "reasoning_steps"):
            reasoning_steps = final_chunk.reasoning_steps or []
        if not citations and hasattr(final_chunk, "citations"):
            citations = final_chunk.citations or []

    response_text = "".join(response_text_parts)

    # Convert search_results to list of dicts if they're objects
    search_results_dicts = []
    for sr in search_results:
        if hasattr(sr, "__dict__"):
            search_results_dicts.append({
                "title": getattr(sr, "title", ""),
                "url": getattr(sr, "url", ""),
                "date": getattr(sr, "date", None),
                "snippet": getattr(sr, "snippet", ""),
                "source": getattr(sr, "source", "web"),
            })
        elif isinstance(sr, dict):
            search_results_dicts.append(sr)

    # Convert reasoning_steps and extract fetch_url_content
    reasoning_steps_dicts = []
    fetched_contents = []  # Collect all fetched URL contents
    for rs in reasoning_steps:
        if hasattr(rs, "__dict__"):
            step_dict = {
                "thought": getattr(rs, "thought", ""),
                "type": getattr(rs, "type", ""),
            }
            if hasattr(rs, "web_search"):
                step_dict["web_search"] = rs.web_search
            if hasattr(rs, "fetch_url_content"):
                fetch_data = rs.fetch_url_content
                step_dict["fetch_url_content"] = fetch_data
                # Extract individual fetched contents
                if hasattr(fetch_data, "contents"):
                    for content in fetch_data.contents:
                        if hasattr(content, "__dict__"):
                            fetched_contents.append({
                                "title": getattr(content, "title", ""),
                                "url": getattr(content, "url", ""),
                                "date": getattr(content, "date", None),
                                "last_updated": getattr(content, "last_updated", None),
                                "snippet": getattr(content, "snippet", ""),
                                "source": getattr(content, "source", "web"),
                            })
                        elif isinstance(content, dict):
                            fetched_contents.append(content)
                elif isinstance(fetch_data, dict) and "contents" in fetch_data:
                    fetched_contents.extend(fetch_data["contents"])
            reasoning_steps_dicts.append(step_dict)
        elif isinstance(rs, dict):
            reasoning_steps_dicts.append(rs)
            # Also extract fetch_url_content from dict format
            if rs.get("type") == "fetch_url_content" and "fetch_url_content" in rs:
                contents = rs["fetch_url_content"].get("contents", [])
                fetched_contents.extend(contents)

    return PerplexityProSearchResult(
        response_text=response_text,
        citations=list(citations) if citations else [],
        search_results=search_results_dicts,
        reasoning_steps=reasoning_steps_dicts,
        fetched_contents=fetched_contents,
    )


async def perplexity_search(
    query: str,
    before_date: datetime | None = None,
    model: str = "sonar-pro",
    use_pro_search: bool = True,
) -> tuple[str, list[str], list[dict], list[dict], list[dict]]:
    """
    Search using Perplexity's Sonar models.

    Args:
        query: The search query
        before_date: Only include results last modified before this date
        model: Perplexity model to use (sonar, sonar-pro)
        use_pro_search: Whether to use Pro Search features (multi-step reasoning)

    Returns:
        Tuple of (response_text, citations, search_results, reasoning_steps, fetched_contents)
        - response_text: The synthesized response
        - citations: List of citation URLs
        - search_results: List of articles with title, url, date, snippet (Pro Search only)
        - reasoning_steps: Multi-step reasoning with tool usage (Pro Search only)
        - fetched_contents: Content fetched via fetch_url_content tool (Pro Search only)
    """
    # Run sync client in thread pool for async compatibility
    result = await asyncio.to_thread(
        _sync_perplexity_search,
        query=query,
        before_date=before_date,
        model=model,
        use_pro_search=use_pro_search,
    )

    return result.response_text, result.citations, result.search_results, result.reasoning_steps, result.fetched_contents


async def check_leakage_with_llm(
    api: InferenceAPI,
    question: str,
    answer: str,
    cutoff_date: datetime,
    search_result: str,
    model_id: str,
) -> dict:
    """
    Use LLM to check if search results leak resolution information.

    Returns dict with:
        - has_leakage: bool
        - confidence: int (0-100)
        - risk_level: str
        - explanation: str
    """
    prompt = f"""You are checking search results for information leakage.

QUESTION: {question}
ACTUAL ANSWER (DO NOT USE THIS - only for leak detection): {answer}
CUTOFF DATE: {cutoff_date.strftime('%Y-%m-%d')} (results should only contain info BEFORE this)

SEARCH RESULTS:
{search_result}

Analyze if these search results contain:
1. Information that directly reveals the answer
2. Events/dates that occurred AFTER the cutoff date
3. Any indication of the final resolution

Return JSON only (no markdown):
{{
    "has_leakage": true/false,
    "confidence": 0-100,
    "risk_level": "none|low|medium|high|critical",
    "leaked_info": ["list of specific leaks if any"],
    "explanation": "brief explanation"
}}"""

    response = await ask_single_question(
        api=api,
        model_id=model_id,
        question=prompt,
        system_prompt="You are an information leak detector. Return valid JSON only.",
    )

    try:
        return json.loads(response[0])
    except json.JSONDecodeError:
        return {
            "has_leakage": True,  # Assume leakage if we can't parse
            "confidence": 0,
            "risk_level": "unknown",
            "leaked_info": [],
            "explanation": f"Failed to parse LLM response: {response[0][:200]}",
        }


async def generate_background_prompt(
    api: InferenceAPI,
    question_title: str,
    resolution_criteria: str,
    cutoff_date: datetime,
    model_id: str,
) -> str:
    """
    Generate a vague background-focused search prompt that avoids searching for outcomes.

    Instead of asking about specific predictions/resolutions, generates prompts that
    gather contextual background information useful for making a forecast.
    """
    prompt = f"""You are helping create a web search query. The goal is to gather BACKGROUND INFORMATION
that would help someone make an informed prediction about a future event.

CRITICAL: The search query must NOT:
- Ask about outcomes, results, or resolutions
- Mention predictions or forecasts
- Ask "will X happen" or "did X happen"
- Search for specific dates of future events
- Look for news about how something turned out

The search query SHOULD:
- Ask about historical context and trends
- Request information about key actors, stakeholders, or organizations involved
- Gather information about relevant policies, rules, or procedures
- Look for expert analysis or background reporting
- Focus on understanding the situation as of {cutoff_date.strftime('%B %Y')}

MARKET QUESTION: {question_title}
CONTEXT: {resolution_criteria[:500] if resolution_criteria else 'N/A'}

Generate a single search query (1-2 sentences) that gathers useful background information
without revealing or searching for the outcome. Return ONLY the search query, nothing else."""

    response = await ask_single_question(
        api=api,
        model_id=model_id,
        question=prompt,
        system_prompt="You generate web search queries. Return only the search query text, no quotes or explanation.",
    )

    return response[0].strip().strip('"').strip("'")


def _fetch_article_content(url: str) -> tuple[str, str | None]:
    """
    Fetch article content using trafilatura (sync, called from thread pool).

    Returns:
        Tuple of (url, content) where content is None if fetch failed.
    """
    # Skip PDFs
    if url.lower().endswith(".pdf"):
        return url, None

    # Skip YouTube links
    if "youtube.com" in url.lower() or "youtu.be" in url.lower():
        return url, None

    # Skip known JS-heavy sites that don't work with trafilatura
    skip_domains = [
        "rottentomatoes.com",
        "imdb.com",
        "twitter.com",
        "x.com",
        "facebook.com",
        "instagram.com",
        "tiktok.com",
        "linkedin.com",
    ]
    url_lower = url.lower()
    for domain in skip_domains:
        if domain in url_lower:
            return url, None

    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            content = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
            )
            return url, content
        return url, None
    except Exception as e:
        LOGGER.warning(f"Failed to fetch {url}: {e}")
        return url, None


_PROCESS_POOL: ProcessPoolExecutor | None = None


def _get_process_pool(max_workers: int = 50) -> ProcessPoolExecutor:
    """Get or create a shared ProcessPoolExecutor."""
    global _PROCESS_POOL
    if _PROCESS_POOL is None:
        _PROCESS_POOL = ProcessPoolExecutor(max_workers=max_workers)
    return _PROCESS_POOL


async def fetch_articles_parallel(
    urls: list[str],
    max_workers: int = 50,
) -> dict[str, str | None]:
    """
    Fetch multiple articles using trafilatura with a shared ProcessPoolExecutor.

    Uses processes instead of threads because trafilatura/lxml are not thread-safe
    and can cause malloc corruption with ThreadPoolExecutor.

    Returns:
        Dict mapping URL -> content (None if fetch failed)
    """
    loop = asyncio.get_event_loop()
    executor = _get_process_pool(max_workers)

    results = await asyncio.gather(*[
        loop.run_in_executor(executor, _fetch_article_content, url)
        for url in urls
    ])

    return dict(results)


async def check_article_leakage(
    api: InferenceAPI,
    question: str,
    answer: str,
    cutoff_date: datetime,
    article_url: str,
    article_title: str,
    article_date: str | None,
    article_content: str,
    model_id: str,
    max_content_length: int = 10000,
) -> dict:
    """
    Check if a single article contains information leakage.

    Returns dict with:
        - url: str
        - title: str
        - has_leakage: bool
        - confidence: int (0-100)
        - explanation: str
        - content_preview: str (first 500 chars)
    """
    # Truncate content if too long
    truncated_content = article_content[:max_content_length]
    if len(article_content) > max_content_length:
        truncated_content += f"\n... [truncated, {len(article_content)} total chars]"

    article_date_str = article_date if article_date else "Unknown"

    prompt = f"""You are checking if an article contains information that would leak the answer to a prediction market question.

PREDICTION QUESTION: {question}
ACTUAL ANSWER (DO NOT USE THIS - only for leak detection): {answer}
CUTOFF DATE: {cutoff_date.strftime('%Y-%m-%d')}

ARTICLE TITLE: {article_title}
ARTICLE URL: {article_url}
ARTICLE PUBLICATION DATE: {article_date_str}
ARTICLE CONTENT:
{truncated_content}

CRITICAL RULES FOR LEAKAGE DETECTION:

1. LEAKAGE = information about events that occurred AFTER the cutoff date ({cutoff_date.strftime('%Y-%m-%d')}) that reveals how the prediction resolved.

2. NOT LEAKAGE: Events, news, or information from BEFORE the cutoff date are NOT leakage, even if they seem related to or predictive of the outcome. Historical context, prior events, trends, and background information from before the cutoff are VALID and USEFUL for forecasting.

3. Only mark as leakage if the article contains:
   - Explicit statement of the outcome/resolution
   - News of events dated AFTER the cutoff that directly resolve the question
   - Clear post-cutoff information that reveals the answer

4. Note: The article publication date ({article_date_str}) is provided for context, but websites can update articles after publication. Focus on the CONTENT - look for mentions of specific dates or events that occurred after the cutoff.

Return JSON only (no markdown):
{{
    "has_leakage": true/false,
    "confidence": 0-100,
    "explanation": "brief explanation"
}}"""

    response = await ask_single_question(
        api=api,
        model_id=model_id,
        question=prompt,
        system_prompt="You are an information leak detector for prediction markets. Return valid JSON only.",
    )

    try:
        result = json.loads(response[0])
    except json.JSONDecodeError:
        result = {
            "has_leakage": True,  # Assume leakage if we can't parse
            "confidence": 0,
            "explanation": f"Failed to parse LLM response: {response[0][:200]}",
        }

    return {
        "url": article_url,
        "title": article_title,
        "has_leakage": result.get("has_leakage", True),
        "confidence": result.get("confidence", 0),
        "explanation": result.get("explanation", ""),
        "content_length": len(article_content),
        "content_preview": article_content[:500],
    }


async def score_article_relevance(
    api: InferenceAPI,
    question: str,
    resolution_criteria: str,
    article_url: str,
    article_title: str,
    article_content: str,
    model_id: str,
    max_content_length: int = 10000,
) -> dict:
    """
    Score how relevant an article is for answering a prediction market question.

    Returns dict with:
        - relevance_score: int (0-100)
        - relevance_category: str (high/medium/low/none)
        - useful_info: list of key facts from article
        - explanation: str
    """
    truncated_content = article_content[:max_content_length]
    if len(article_content) > max_content_length:
        truncated_content += f"\n... [truncated, {len(article_content)} total chars]"

    prompt = f"""You are evaluating how relevant an article is for making a forecast on a prediction market question.

PREDICTION QUESTION: {question}

RESOLUTION CRITERIA: {resolution_criteria if resolution_criteria else "Not specified"}

ARTICLE TITLE: {article_title}
ARTICLE URL: {article_url}
ARTICLE CONTENT:
{truncated_content}

TASK: Evaluate how useful this article is for a forecaster trying to predict the outcome of this question.

Consider:
1. Does the article provide DIRECT information about the subject of the prediction?
2. Does it contain relevant background, context, or historical data?
3. Does it discuss factors that could influence the outcome?
4. Does it contain expert opinions, analysis, or data relevant to the prediction?
5. Is the information specific enough to be actionable for forecasting?

RELEVANCE CATEGORIES:
- "high": Directly about the prediction subject, contains specific facts/data/analysis useful for forecasting
- "medium": Related to the prediction topic, provides useful context or background
- "low": Tangentially related, minimal useful information
- "none": Unrelated or too generic to be useful

Return JSON only (no markdown):
{{
    "relevance_score": 0-100,
    "relevance_category": "high|medium|low|none",
    "useful_info": ["list of 1-3 key facts or insights from this article that are useful for forecasting, or empty if none"],
    "explanation": "brief explanation of relevance assessment"
}}"""

    response = await ask_single_question(
        api=api,
        model_id=model_id,
        question=prompt,
        system_prompt="You are an expert forecaster evaluating information relevance. Return valid JSON only.",
    )

    try:
        result = json.loads(response[0])
    except json.JSONDecodeError:
        result = {
            "relevance_score": 0,
            "relevance_category": "none",
            "useful_info": [],
            "explanation": f"Failed to parse LLM response: {response[0][:200]}",
        }

    return {
        "url": article_url,
        "title": article_title,
        "relevance_score": result.get("relevance_score", 0),
        "relevance_category": result.get("relevance_category", "none"),
        "useful_info": result.get("useful_info", []),
        "explanation": result.get("explanation", ""),
    }


async def run_single_search_and_check(
    api: InferenceAPI,
    query: str,
    answer: str,
    cutoff_date: datetime,
    config: SearchAugmentConfig,
    search_idx: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run a single search and leakage check. Returns result dict."""
    async with semaphore:  # Rate limit Perplexity calls
        try:
            response_text, citations, search_results, reasoning_steps, fetched_contents = await perplexity_search(
                query=query,
                before_date=cutoff_date,
                model=config.perplexity_model,
                use_pro_search=config.use_pro_search,
            )

            leakage_check = await check_leakage_with_llm(
                api=api,
                question=query,
                answer=answer,
                cutoff_date=cutoff_date,
                search_result=response_text,
                model_id=config.judge_model,
            )

            has_leakage = leakage_check.get("has_leakage", True)

            return {
                "search_idx": search_idx,
                "response": response_text,
                "citations": citations,
                "search_results": search_results,  # Pro Search articles
                "reasoning_steps": reasoning_steps,  # Pro Search multi-step reasoning
                "fetched_contents": fetched_contents,  # Content from fetch_url_content
                "leakage_check": leakage_check,
                "has_leakage": has_leakage,
                "error": None,
            }
        except Exception as e:
            LOGGER.error(f"Search {search_idx} failed: {e}")
            return {
                "search_idx": search_idx,
                "response": None,
                "citations": [],
                "search_results": [],
                "reasoning_steps": [],
                "fetched_contents": [],
                "leakage_check": None,
                "has_leakage": None,
                "error": str(e),
            }


async def process_single_market(
    api: InferenceAPI,
    market: dict,
    config: SearchAugmentConfig,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Process a single market with article-based pipeline:
    1. Perplexity search for article URLs
    2. Filter articles by publication date
    3. Fetch full content with trafilatura
    4. Per-article leakage checking
    5. Return validated articles
    """
    question_title = market["question_title"]
    print(f"\n[Market] {question_title[:60]}...")

    # Parse dates
    start_date_str = market["question_start_date"]
    if "T" not in start_date_str:
        question_start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    else:
        question_start_date = datetime.fromisoformat(start_date_str)

    # Calculate cutoff date
    cutoff_date = question_start_date - timedelta(days=config.cutoff_days_before_start)
    print(f"[Cutoff Date] {cutoff_date.strftime('%Y-%m-%d')}")

    # Step 1: Run Perplexity search to discover articles
    if config.use_background_prompt:
        search_prompt = await generate_background_prompt(
            api=api,
            question_title=question_title,
            resolution_criteria=market.get("resolution_criteria", ""),
            cutoff_date=cutoff_date,
            model_id=config.prompt_model,
        )
        print(f"[Background Prompt] {search_prompt}")
    else:
        search_prompt = (
            "Find all the relevant articles and news for the following question.\n"
            f"Question: {question_title}\n"
            f"Resolution Criteria: {market.get('resolution_criteria', '')}"
        )

    async with semaphore:
        response_text, citations, search_results, reasoning_steps, fetched_contents = await perplexity_search(
            query=search_prompt,
            before_date=cutoff_date,
            model=config.perplexity_model,
            use_pro_search=config.use_pro_search,
        )

    # Collect unique article URLs from search_results and citations
    article_metadata = {}  # url -> {title, date, snippet}
    for article in search_results:
        url = article.get("url", "")
        if url and url not in article_metadata:
            article_metadata[url] = {
                "title": article.get("title", ""),
                "date": article.get("date"),
                "snippet": article.get("snippet", ""),
            }

    # Also add citation URLs (may not have metadata)
    for url in citations:
        if url not in article_metadata:
            article_metadata[url] = {"title": "", "date": None, "snippet": ""}

    print(f"[Perplexity] Found {len(article_metadata)} unique article URLs")

    # Step 2: Filter articles by publication date
    articles_before_cutoff = {}
    articles_after_cutoff = []
    articles_no_date = {}

    for url, meta in article_metadata.items():
        date_str = meta.get("date")
        if date_str:
            try:
                # Parse date (format: YYYY-MM-DD)
                article_date = datetime.strptime(date_str, "%Y-%m-%d")
                if article_date < cutoff_date:
                    articles_before_cutoff[url] = meta
                else:
                    articles_after_cutoff.append({"url": url, "date": date_str, **meta})
            except ValueError:
                # Can't parse date, include it but flag
                articles_no_date[url] = meta
        else:
            articles_no_date[url] = meta

    print(f"[Date Filter] Before cutoff: {len(articles_before_cutoff)}, After: {len(articles_after_cutoff)}, No date: {len(articles_no_date)}")

    # Combine articles to fetch (before cutoff + no date)
    articles_to_fetch = {**articles_before_cutoff, **articles_no_date}

    # Limit number of articles
    if len(articles_to_fetch) > config.max_articles_per_market:
        # Prioritize articles with dates
        urls_with_dates = list(articles_before_cutoff.keys())[:config.max_articles_per_market]
        remaining_slots = config.max_articles_per_market - len(urls_with_dates)
        urls_no_dates = list(articles_no_date.keys())[:remaining_slots]
        articles_to_fetch = {url: articles_to_fetch[url] for url in urls_with_dates + urls_no_dates}
        print(f"[Limit] Reduced to {len(articles_to_fetch)} articles (max {config.max_articles_per_market})")

    # Step 3: Fetch full article content with trafilatura
    if articles_to_fetch:
        print(f"[Fetching] {len(articles_to_fetch)} articles with trafilatura...")
        url_to_content = await fetch_articles_parallel(
            urls=list(articles_to_fetch.keys()),
            max_workers=config.article_fetch_concurrency,
        )
        successful_fetches = {url: content for url, content in url_to_content.items() if content}
        print(f"[Fetched] {len(successful_fetches)}/{len(articles_to_fetch)} articles successfully")
    else:
        successful_fetches = {}

    # Step 4: Per-article leakage checking
    validated_articles = []
    leaked_articles = []
    failed_articles = []

    if successful_fetches:
        print(f"[Leakage Check] Checking {len(successful_fetches)} articles...")
        leakage_tasks = [
            check_article_leakage(
                api=api,
                question=question_title,
                answer=market["answer"],
                cutoff_date=cutoff_date,
                article_url=url,
                article_title=articles_to_fetch[url]["title"],
                article_date=articles_to_fetch[url].get("date"),
                article_content=content,
                model_id=config.judge_model,
                max_content_length=config.max_article_length,
            )
            for url, content in successful_fetches.items()
        ]

        leakage_results = await asyncio.gather(*leakage_tasks)

        # Separate leaked vs clean articles
        clean_articles = []
        for result in leakage_results:
            if result["has_leakage"]:
                leaked_articles.append(result)
            else:
                result["content"] = successful_fetches[result["url"]]
                clean_articles.append(result)

        # Step 5: Score relevance for clean articles
        if clean_articles:
            print(f"[Relevance] Scoring {len(clean_articles)} clean articles...")
            relevance_tasks = [
                score_article_relevance(
                    api=api,
                    question=question_title,
                    resolution_criteria=market.get("resolution_criteria", ""),
                    article_url=article["url"],
                    article_title=article["title"],
                    article_content=article["content"],
                    model_id=config.judge_model,
                    max_content_length=config.max_article_length,
                )
                for article in clean_articles
            ]

            relevance_results = await asyncio.gather(*relevance_tasks)

            # Merge relevance scores into validated articles
            for article, relevance in zip(clean_articles, relevance_results):
                article["relevance_score"] = relevance["relevance_score"]
                article["relevance_category"] = relevance["relevance_category"]
                article["useful_info"] = relevance["useful_info"]
                article["relevance_explanation"] = relevance["explanation"]
                validated_articles.append(article)

            # Sort by relevance score (highest first)
            validated_articles.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    # Track failed fetches
    for url in articles_to_fetch:
        if url not in successful_fetches:
            failed_articles.append({
                "url": url,
                "title": articles_to_fetch[url]["title"],
                "reason": "fetch_failed",
            })

    print(f"[Result] Valid: {len(validated_articles)}, Leaked: {len(leaked_articles)}, Failed: {len(failed_articles)}")

    # Calculate article leakage rate
    total_checked = len(validated_articles) + len(leaked_articles)
    article_leakage_rate = len(leaked_articles) / total_checked if total_checked > 0 else 0.0

    # Build search context with article-based results
    search_context = {
        # Pipeline config
        "cutoff_date": cutoff_date.isoformat(),
        "perplexity_model": config.perplexity_model,
        "use_pro_search": config.use_pro_search,
        "use_background_prompt": config.use_background_prompt,
        "search_prompt": search_prompt,

        # Discovery stats
        "num_articles_discovered": len(article_metadata),
        "num_articles_before_cutoff": len(articles_before_cutoff),
        "num_articles_after_cutoff": len(articles_after_cutoff),
        "num_articles_no_date": len(articles_no_date),
        "articles_filtered_by_date": articles_after_cutoff,  # These were excluded

        # Fetch stats
        "num_articles_fetched": len(successful_fetches),
        "num_articles_fetch_failed": len(failed_articles),

        # Leakage stats
        "num_articles_validated": len(validated_articles),
        "num_articles_leaked": len(leaked_articles),
        "article_leakage_rate": round(article_leakage_rate, 4),

        # The good stuff: validated articles with full content
        "validated_articles": validated_articles,  # Clean articles for RAG

        # Details for debugging
        "leaked_articles": leaked_articles,  # Articles that leaked (no content, just metadata)
        "failed_articles": failed_articles,  # Articles that failed to fetch

        # Perplexity raw response (for reference)
        "perplexity_response": response_text,
        "perplexity_citations": citations,
    }

    return {**market, "search_context": search_context}


async def process_and_save(
    api: InferenceAPI,
    market: dict,
    config: SearchAugmentConfig,
    semaphore: asyncio.Semaphore,
    output_file,
    file_lock: asyncio.Lock,
) -> dict:
    """Process a market and immediately write result to file."""
    result = await process_single_market(api, market, config, semaphore)

    # Write to file immediately with lock
    async with file_lock:
        output_file.write(json.dumps(result) + "\n")
        output_file.flush()

    return result


async def main():
    parser = ArgumentParser()
    parser.add_arguments(SearchAugmentConfig, dest="config")
    args = parser.parse_args()
    config: SearchAugmentConfig = args.config

    config.setup_experiment(log_file_prefix="search_augmented")

    output_path = Path(config.output_dir) / "search_augmented_markets.jsonl"

    # Load all markets from dataset
    all_markets = []
    with open(config.dataset_path) as f:
        for line in f:
            all_markets.append(json.loads(line))

    # Check for already-processed markets (resume support)
    processed_ids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Use (question_title, resolution_criteria) as ID for uniqueness
                    processed_ids.add((data.get("question_title"), data.get("resolution_criteria")))
                except json.JSONDecodeError:
                    continue
        if processed_ids:
            LOGGER.info(f"Resuming: found {len(processed_ids)} already-processed markets")

    # Randomly sample num_tasks markets (using seed for reproducibility)
    if config.num_tasks < len(all_markets):
        random.seed(config.seed)
        markets = random.sample(all_markets, config.num_tasks)
        LOGGER.info(f"Randomly selected {len(markets)} markets from {len(all_markets)} total (seed={config.seed})")
    else:
        markets = all_markets
        LOGGER.info(f"Loaded all {len(markets)} markets from {config.dataset_path}")

    # Filter out already-processed markets
    markets_to_process = [m for m in markets if (m.get("question_title"), m.get("resolution_criteria")) not in processed_ids]
    if len(markets_to_process) < len(markets):
        LOGGER.info(f"Skipping {len(markets) - len(markets_to_process)} already-processed markets")

    # Initialize API
    api = InferenceAPI()

    # Create semaphores for rate limiting
    perplexity_semaphore = asyncio.Semaphore(config.perplexity_concurrency)
    market_semaphore = asyncio.Semaphore(50)  # Max concurrent markets
    file_lock = asyncio.Lock()

    async def process_with_limit(market):
        async with market_semaphore:
            return await process_and_save(api, market, config, perplexity_semaphore, output_file, file_lock)

    # Process markets
    search_mode_log = "Pro Search" if config.use_pro_search else "Basic Sonar"
    prompt_mode_log = "background prompts" if config.use_background_prompt else "direct prompts"
    LOGGER.info(
        f"Processing {len(markets_to_process)} markets "
        f"(max 25 concurrent markets, {config.perplexity_concurrency} concurrent Perplexity calls, "
        f"{search_mode_log}, {prompt_mode_log})"
    )

    # Open file in append mode and process with live saving
    with open(output_path, "a") as output_file:
        tasks = [
            process_with_limit(market)
            for market in markets_to_process
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Processing markets")

    # Calculate stats for article-based pipeline
    total_discovered = sum(r["search_context"]["num_articles_discovered"] for r in results)
    total_fetched = sum(r["search_context"]["num_articles_fetched"] for r in results)
    total_validated = sum(r["search_context"]["num_articles_validated"] for r in results)
    total_leaked = sum(r["search_context"]["num_articles_leaked"] for r in results)
    total_failed = sum(r["search_context"]["num_articles_fetch_failed"] for r in results)
    total_filtered_by_date = sum(r["search_context"]["num_articles_after_cutoff"] for r in results)

    total_checked = total_validated + total_leaked
    avg_leakage_rate = total_leaked / total_checked if total_checked > 0 else 0.0

    LOGGER.info(f"Discovered: {total_discovered}, Fetched: {total_fetched}, Validated: {total_validated}, Leaked: {total_leaked}")

    search_mode = "Pro Search" if config.use_pro_search else "Basic Sonar"
    prompt_mode = "Background (LLM-generated)" if config.use_background_prompt else "Direct"

    print(f"\n{'='*70}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*70}")
    print(f"Markets processed: {len(results)}")
    print(f"Perplexity model: {config.perplexity_model} ({search_mode})")
    print(f"Prompt mode: {prompt_mode}")
    print(f"{'='*70}")
    print("ARTICLE PIPELINE STATS:")
    print(f"  Articles discovered (Perplexity): {total_discovered}")
    print(f"  Articles filtered by date:       {total_filtered_by_date}")
    print(f"  Articles fetched (trafilatura):  {total_fetched}")
    print(f"  Articles fetch failed:           {total_failed}")
    print(f"{'='*70}")
    print("LEAKAGE STATS (per-article):")
    print(f"  Articles validated (clean):      {total_validated}")
    print(f"  Articles leaked:                 {total_leaked}")
    print(f"  Article leakage rate:            {avg_leakage_rate:.1%}")
    print(f"{'='*70}")
    print("\nPer-market breakdown:")
    for r in results:
        ctx = r["search_context"]
        q = r["question_title"][:50]
        v = ctx["num_articles_validated"]
        l = ctx["num_articles_leaked"]
        rate = ctx["article_leakage_rate"]
        print(f"  {q}... : {v} valid, {l} leaked ({rate:.0%})")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
