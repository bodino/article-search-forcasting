#!/usr/bin/env python
"""
Simple web server to review articles for information leakage.

Shows forecast questions with their top 3 articles, allowing manual
annotation of which articles leak information.

Usage:
    uv run leakage_reviewer.py --data_path search_augmented_data.jsonl --port 8765
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, render_template_string, request, jsonify
from simple_parsing import ArgumentParser

app = Flask(__name__)

# Global state
DATA = []
ANNOTATIONS = {}
CONFIG = None


@dataclass
class ReviewerConfig:
    data_path: str = field(metadata={"help": "Path to search-augmented JSONL file"})
    query_field: str = field(default="question_title", metadata={"help": "Field containing the query/question"})
    answer_field: str = field(default="answer", metadata={"help": "Field containing the ground truth answer"})
    num_samples: int = field(default=100, metadata={"help": "Number of random samples to review"})
    output_path: str = field(default="leakage_annotations.json", metadata={"help": "Path to save annotations"})
    port: int = field(default=8765, metadata={"help": "Port to run server on"})
    seed: int = field(default=42, metadata={"help": "Random seed for sampling"})


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Leakage Reviewer - Question {{ current + 1 }}/{{ total }}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .question {
            font-size: 1.3em;
            font-weight: 600;
            color: #333;
            margin-bottom: 10px;
        }
        .meta {
            display: flex;
            gap: 20px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .meta-item {
            background: #e8e8e8;
            padding: 5px 12px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .meta-item.answer {
            background: #d4edda;
            color: #155724;
            font-weight: 600;
        }
        .meta-item.cutoff {
            background: #fff3cd;
            color: #856404;
        }
        .article {
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .article-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 15px;
        }
        .article-title {
            font-weight: 600;
            color: #0066cc;
            text-decoration: none;
            flex: 1;
        }
        .article-title:hover {
            text-decoration: underline;
        }
        .article-content {
            margin-top: 10px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
            font-size: 0.9em;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        .leakage-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.2s;
        }
        .leakage-btn.leaks {
            background: #dc3545;
            color: white;
        }
        .leakage-btn.safe {
            background: #28a745;
            color: white;
        }
        .leakage-btn.unmarked {
            background: #6c757d;
            color: white;
        }
        .leakage-btn:hover {
            opacity: 0.9;
            transform: scale(1.02);
        }
        .nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
        }
        .nav-btn {
            padding: 12px 24px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            text-decoration: none;
        }
        .nav-btn:hover {
            background: #0056b3;
        }
        .nav-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .progress {
            text-align: center;
            color: #666;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: #28a745;
            border-radius: 4px;
            transition: width 0.3s;
        }
        .relevance {
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }
        .relevance-high { color: #28a745; }
        .relevance-medium { color: #ffc107; }
        .relevance-low { color: #dc3545; }
        .no-articles {
            color: #666;
            font-style: italic;
            padding: 20px;
            text-align: center;
        }
        .stats {
            background: #e8f4f8;
            padding: 10px 15px;
            border-radius: 4px;
            margin-bottom: 15px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="stats">
            Annotated: {{ annotated }}/{{ total }} |
            Leaking articles found: {{ total_leaks }}
        </div>

        <div class="question">{{ question }}</div>

        <div class="meta">
            <span class="meta-item answer">Answer: {{ answer }}</span>
            <span class="meta-item cutoff">Cutoff: {{ cutoff_date }}</span>
            <span class="meta-item">{{ num_articles }} articles found</span>
        </div>
    </div>

    <h3>Top Articles (mark if they leak the answer)</h3>

    {% if articles %}
        {% for article in articles %}
        <div class="article" id="article-{{ loop.index0 }}">
            <div class="article-header">
                <div>
                    <a href="{{ article.url }}" target="_blank" class="article-title">
                        {{ article.title or article.url }}
                    </a>
                    <div class="relevance relevance-{{ article.relevance_category or 'low' }}">
                        Relevance: {{ article.relevance_score or 'N/A' }}% ({{ article.relevance_category or 'unknown' }})
                    </div>
                </div>
                <button class="leakage-btn {{ 'leaks' if annotations.get(loop.index0) == 'leaks' else ('safe' if annotations.get(loop.index0) == 'safe' else 'unmarked') }}"
                        onclick="markArticle({{ loop.index0 }})">
                    {{ 'LEAKS' if annotations.get(loop.index0) == 'leaks' else ('SAFE' if annotations.get(loop.index0) == 'safe' else 'Click to mark') }}
                </button>
            </div>
            <div class="article-content">{{ article.content_preview or article.content[:1000] if article.content else 'No content available' }}</div>
        </div>
        {% endfor %}
    {% else %}
        <div class="card no-articles">No articles found for this question</div>
    {% endif %}

    <div class="nav">
        <a href="/question/{{ current - 1 }}" class="nav-btn" {% if current == 0 %}style="visibility: hidden"{% endif %}>&larr; Previous</a>
        <div class="progress">
            Question {{ current + 1 }} of {{ total }}
            <div class="progress-bar">
                <div class="progress-fill" style="width: {{ (annotated / total * 100) | int }}%"></div>
            </div>
        </div>
        <a href="/question/{{ current + 1 }}" class="nav-btn" {% if current >= total - 1 %}style="visibility: hidden"{% endif %}>Next &rarr;</a>
    </div>

    <script>
        function markArticle(articleIdx) {
            const btn = document.querySelector(`#article-${articleIdx} .leakage-btn`);
            let currentState = btn.textContent.trim();
            let newState;

            if (currentState === 'Click to mark' || currentState === 'SAFE') {
                newState = 'leaks';
            } else {
                newState = 'safe';
            }

            fetch('/annotate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    question_idx: {{ current }},
                    article_idx: articleIdx,
                    annotation: newState
                })
            }).then(response => response.json())
              .then(data => {
                  btn.textContent = newState === 'leaks' ? 'LEAKS' : 'SAFE';
                  btn.className = `leakage-btn ${newState === 'leaks' ? 'leaks' : 'safe'}`;
                  // Update stats
                  location.reload();
              });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return f'<script>window.location.href="/question/0";</script>'


@app.route('/question/<int:idx>')
def question(idx):
    global DATA, ANNOTATIONS, CONFIG

    if idx < 0 or idx >= len(DATA):
        return "Question not found", 404

    item = DATA[idx]
    ctx = item.get("search_context", {})
    articles = ctx.get("validated_articles", [])[:3]  # Top 3 only

    # Get annotations for this question
    q_annotations = ANNOTATIONS.get(str(idx), {})

    # Count stats
    annotated = sum(1 for q in ANNOTATIONS.values() if q)
    total_leaks = sum(
        1 for q in ANNOTATIONS.values()
        for a in q.values() if a == 'leaks'
    )

    return render_template_string(
        HTML_TEMPLATE,
        current=idx,
        total=len(DATA),
        question=item.get(CONFIG.query_field, "No question"),
        answer=item.get(CONFIG.answer_field, "Unknown"),
        cutoff_date=ctx.get("cutoff_date", "Unknown")[:10],
        num_articles=ctx.get("num_articles_validated", len(articles)),
        articles=articles,
        annotations=q_annotations,
        annotated=annotated,
        total_leaks=total_leaks
    )


@app.route('/annotate', methods=['POST'])
def annotate():
    global ANNOTATIONS, CONFIG

    data = request.json
    q_idx = str(data['question_idx'])
    a_idx = str(data['article_idx'])
    annotation = data['annotation']

    if q_idx not in ANNOTATIONS:
        ANNOTATIONS[q_idx] = {}
    ANNOTATIONS[q_idx][a_idx] = annotation

    # Save to file
    with open(CONFIG.output_path, 'w') as f:
        json.dump(ANNOTATIONS, f, indent=2)

    return jsonify({"status": "ok"})


@app.route('/export')
def export():
    """Export annotations as downloadable JSON."""
    return jsonify(ANNOTATIONS)


def main():
    global DATA, ANNOTATIONS, CONFIG

    parser = ArgumentParser()
    parser.add_arguments(ReviewerConfig, dest="config")
    args = parser.parse_args()
    CONFIG = args.config

    # Load data
    all_data = []
    with open(CONFIG.data_path) as f:
        for line in f:
            all_data.append(json.loads(line))

    # Sample random subset
    random.seed(CONFIG.seed)
    if CONFIG.num_samples < len(all_data):
        DATA = random.sample(all_data, CONFIG.num_samples)
    else:
        DATA = all_data

    print(f"Loaded {len(DATA)} questions for review")

    # Load existing annotations if they exist
    if Path(CONFIG.output_path).exists():
        with open(CONFIG.output_path) as f:
            ANNOTATIONS = json.load(f)
        print(f"Loaded {len(ANNOTATIONS)} existing annotations from {CONFIG.output_path}")

    print(f"\nStarting server at http://localhost:{CONFIG.port}")
    print(f"Annotations will be saved to: {CONFIG.output_path}")

    app.run(host='0.0.0.0', port=CONFIG.port, debug=False)


if __name__ == "__main__":
    main()
