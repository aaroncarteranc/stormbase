"""
Example: Autonomous KB Builder using a local or API-hosted LLM + DuckDuckGo search.

Usage:
    python examples/kb_builder_examples/run_kb_builder.py \
        --topic "US-Iran Relations" \
        --output-dir ./kb_output \
        --model ollama/llama3:70b

The output directory will be an Obsidian vault with three tiers:
  original-documents/  — full-page markdown fetches of every source URL
  indexed/             — AI-synthesized articles with wikilinks
  discussion/          — cross-cutting thematic analysis documents
  _meta/               — kb-index.md, run-log.md, queue.json (crash recovery)
"""

import argparse
import os

from knowledge_storm.collaborative_storm.engine import CollaborativeStormLMConfigs
from knowledge_storm.kb_builder import KBBuilderConfig, KBOrchestrator
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import DuckDuckGoSearchRM


def main():
    parser = argparse.ArgumentParser(
        description="KB Builder — autonomous Obsidian KB generation"
    )
    parser.add_argument("--topic", required=True, help="Broad topic for the knowledge base")
    parser.add_argument("--output-dir", required=True, help="Output directory (Obsidian vault)")
    parser.add_argument(
        "--model",
        default="ollama/llama3:70b",
        help="LiteLLM model string (e.g. ollama/llama3:70b, together_ai/meta-llama/...)",
    )
    parser.add_argument("--max-articles", type=int, default=50)
    parser.add_argument(
        "--max-ceiling", type=int, default=40, help="Max conversation turns per article"
    )
    parser.add_argument(
        "--min-floor", type=int, default=10, help="Min turns before completion checks begin"
    )
    args = parser.parse_args()

    lm_kwargs = {"model": args.model, "max_tokens": 2000, "model_type": "chat"}

    lm_config = CollaborativeStormLMConfigs()
    lm_config.set_question_answering_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 1000}))
    lm_config.set_discourse_manage_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 500}))
    lm_config.set_utterance_polishing_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 2000}))
    lm_config.set_warmstart_outline_gen_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 500}))
    lm_config.set_question_asking_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 300}))
    lm_config.set_knowledge_base_lm(LitellmModel(**{**lm_kwargs, "max_tokens": 1000}))

    rm = DuckDuckGoSearchRM(k=10)

    config = KBBuilderConfig(
        topic=args.topic,
        output_dir=args.output_dir,
        max_articles=args.max_articles,
        max_ceiling=args.max_ceiling,
        min_floor=args.min_floor,
    )

    print(f"Starting KB Builder")
    print(f"  Topic:  {args.topic!r}")
    print(f"  Vault:  {args.output_dir}")
    print(f"  Model:  {args.model}")
    print(f"  Max articles: {args.max_articles}")

    orchestrator = KBOrchestrator(config=config, lm_config=lm_config, rm=rm)
    orchestrator.run()

    print(f"\nDone. Open {args.output_dir!r} in Obsidian.")


if __name__ == "__main__":
    main()
