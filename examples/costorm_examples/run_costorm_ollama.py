"""
Co-STORM pipeline powered by a local model hosted by Ollama and a search engine of your choice.
No API keys are required for the LM. For retrieval, DuckDuckGo requires no key; others need keys in secrets.toml.

Usage:
    python examples/costorm_examples/run_costorm_ollama.py \
        --model gemma4:e4b \
        --retriever duckduckgo \
        --enable_log_print

Output will be structured as below
args.output_dir/
    report.md             # Final article generated
    instance_dump.json    # Full runner state
    log.json              # Log of information-seeking conversation
"""

import json
import os
import traceback
from argparse import ArgumentParser

from knowledge_storm.collaborative_storm.engine import (
    CollaborativeStormLMConfigs,
    CoStormRunner,
    RunnerArgument,
)
from knowledge_storm.collaborative_storm.modules.callback import (
    LocalConsolePrintCallBackHandler,
)
from knowledge_storm.lm import OllamaClient
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.rm import (
    BingSearch,
    BraveRM,
    DuckDuckGoSearchRM,
    SerperRM,
    TavilySearchRM,
    YouRM,
    SearXNG,
)
from knowledge_storm.utils import load_api_key


def main(args):
    load_api_key(toml_file_path="secrets.toml")

    ollama_kwargs = {
        "model": args.model,
        "port": args.port,
        "url": args.url,
        "stop": ("\n\n---",),  # dspy uses "\n\n---" to separate examples
    }

    lm_config: CollaborativeStormLMConfigs = CollaborativeStormLMConfigs()
    lm_config.set_question_answering_lm(OllamaClient(max_tokens=10000, **ollama_kwargs))
    lm_config.set_discourse_manage_lm(OllamaClient(max_tokens=5000, **ollama_kwargs))
    lm_config.set_utterance_polishing_lm(OllamaClient(max_tokens=20000, **ollama_kwargs))
    lm_config.set_warmstart_outline_gen_lm(OllamaClient(max_tokens=5000, **ollama_kwargs))
    lm_config.set_question_asking_lm(OllamaClient(max_tokens=3000, **ollama_kwargs))
    lm_config.set_knowledge_base_lm(OllamaClient(max_tokens=10000, **ollama_kwargs))

    topic = input("Topic: ")
    runner_argument = RunnerArgument(
        topic=topic,
        retrieve_top_k=args.retrieve_top_k,
        max_search_queries=args.max_search_queries,
        total_conv_turn=args.total_conv_turn,
        max_search_thread=args.max_search_thread,
        max_search_queries_per_turn=args.max_search_queries_per_turn,
        warmstart_max_num_experts=args.warmstart_max_num_experts,
        warmstart_max_turn_per_experts=args.warmstart_max_turn_per_experts,
        warmstart_max_thread=args.warmstart_max_thread,
        max_thread_num=args.max_thread_num,
        max_num_round_table_experts=args.max_num_round_table_experts,
        moderator_override_N_consecutive_answering_turn=args.moderator_override_N_consecutive_answering_turn,
        node_expansion_trigger_count=args.node_expansion_trigger_count,
    )

    logging_wrapper = LoggingWrapper(lm_config)
    callback_handler = LocalConsolePrintCallBackHandler() if args.enable_log_print else None

    match args.retriever:
        case "bing":
            rm = BingSearch(
                bing_search_api=os.getenv("BING_SEARCH_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case "you":
            rm = YouRM(
                ydc_api_key=os.getenv("YDC_API_KEY"), k=runner_argument.retrieve_top_k
            )
        case "brave":
            rm = BraveRM(
                brave_search_api_key=os.getenv("BRAVE_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case "duckduckgo":
            rm = DuckDuckGoSearchRM(
                k=runner_argument.retrieve_top_k, safe_search="On", region="us-en"
            )
        case "serper":
            rm = SerperRM(
                serper_search_api_key=os.getenv("SERPER_API_KEY"),
                query_params={"autocorrect": True, "num": 10, "page": 1},
            )
        case "tavily":
            rm = TavilySearchRM(
                tavily_search_api_key=os.getenv("TAVILY_API_KEY"),
                k=runner_argument.retrieve_top_k,
                include_raw_content=True,
            )
        case "searxng":
            rm = SearXNG(
                searxng_api_key=os.getenv("SEARXNG_API_KEY"),
                k=runner_argument.retrieve_top_k,
            )
        case _:
            raise ValueError(
                f'Invalid retriever: {args.retriever}. '
                'Choose from: "bing", "you", "brave", "duckduckgo", "serper", "tavily", "searxng"'
            )

    os.makedirs(args.output_dir, exist_ok=True)

    def save_checkpoint(label="checkpoint"):
        """Write current runner state to disk without disrupting the run."""
        try:
            with open(os.path.join(args.output_dir, "conversation_history.json"), "w") as f:
                json.dump(
                    [t.to_dict() for t in costorm_runner.conversation_history],
                    f, indent=2,
                )
            with open(os.path.join(args.output_dir, "knowledge_base_snapshot.json"), "w") as f:
                json.dump(costorm_runner.knowledge_base.to_dict(), f, indent=2)
            # dump_logging_and_reset with reset_logging=False so the run continues normally
            with open(os.path.join(args.output_dir, "llm_logs.json"), "w") as f:
                json.dump(
                    costorm_runner.logging_wrapper.dump_logging_and_reset(reset_logging=False),
                    f, indent=2,
                )
            print(f"[checkpoint] {label} saved to {args.output_dir}")
        except Exception as e:
            print(f"[checkpoint] failed to save ({label}): {e}")

    costorm_runner = CoStormRunner(
        lm_config=lm_config,
        runner_argument=runner_argument,
        logging_wrapper=logging_wrapper,
        rm=rm,
        callback_handler=callback_handler,
    )

    simulate_user_intent = args.simulate_user_intent or topic

    try:
        # Warm start: builds the initial shared conceptual space / mind map
        costorm_runner.warm_start()
        save_checkpoint("after warm_start")

        # Run the full round-table conversation
        for turn_num in range(1, runner_argument.total_conv_turn + 1):
            print(f"\n--- Turn {turn_num}/{runner_argument.total_conv_turn} ---")

            # System agent turn
            conv_turn = costorm_runner.step()
            if conv_turn is not None:
                print(f"**{conv_turn.role}**: {conv_turn.utterance}\n")
            save_checkpoint(f"turn {turn_num} system")

            # Simulated user turn
            user_turn = costorm_runner.step(
                simulate_user=True,
                simulate_user_intent=simulate_user_intent,
            )
            if user_turn is not None:
                print(f"**{user_turn.role}**: {user_turn.utterance}\n")
            save_checkpoint(f"turn {turn_num} user")

        # Generate and save report
        print("\nRound-table complete. Generating report...")
        costorm_runner.knowledge_base.reorganize()
        article = costorm_runner.generate_report()

        with open(os.path.join(args.output_dir, "report.md"), "w") as f:
            f.write(article)

        with open(os.path.join(args.output_dir, "instance_dump.json"), "w") as f:
            json.dump(costorm_runner.to_dict(), f, indent=2)

        with open(os.path.join(args.output_dir, "log.json"), "w") as f:
            json.dump(costorm_runner.dump_logging_and_reset(), f, indent=2)

        print(f"\nReport saved to {os.path.join(args.output_dir, 'report.md')}")

    except Exception:
        print("\n[error] Run failed — saving emergency checkpoint.")
        with open(os.path.join(args.output_dir, "error.txt"), "w") as f:
            f.write(traceback.format_exc())
        save_checkpoint("on error")
        raise


if __name__ == "__main__":
    parser = ArgumentParser()

    # Ollama server settings
    parser.add_argument("--url", type=str, default="http://localhost", help="Ollama server URL.")
    parser.add_argument("--port", type=int, default=11434, help="Ollama server port.")
    parser.add_argument("--model", type=str, default="llama3:latest", help="Ollama model tag.")

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="./results/co-storm-ollama",
        help="Directory to store the outputs.",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="duckduckgo",
        choices=["bing", "you", "brave", "serper", "duckduckgo", "tavily", "searxng"],
        help="Search engine to use for retrieval.",
    )

    # Co-STORM hyperparameters
    parser.add_argument("--retrieve_top_k", type=int, default=10)
    parser.add_argument("--max_search_queries", type=int, default=2)
    parser.add_argument("--total_conv_turn", type=int, default=20)
    parser.add_argument("--max_search_thread", type=int, default=3,
                        help="Reduce from default 5 to avoid overwhelming a local LM.")
    parser.add_argument("--max_search_queries_per_turn", type=int, default=3)
    parser.add_argument("--warmstart_max_num_experts", type=int, default=3)
    parser.add_argument("--warmstart_max_turn_per_experts", type=int, default=2)
    parser.add_argument("--warmstart_max_thread", type=int, default=2,
                        help="Reduce from default 3 to avoid overwhelming a local LM.")
    parser.add_argument("--max_thread_num", type=int, default=3,
                        help="Reduce from default 10 to avoid overwhelming a local LM.")
    parser.add_argument("--max_num_round_table_experts", type=int, default=2)
    parser.add_argument("--moderator_override_N_consecutive_answering_turn", type=int, default=3)
    parser.add_argument("--node_expansion_trigger_count", type=int, default=10)
    parser.add_argument("--enable_log_print", action="store_true",
                        help="Print conversation turns to console as they happen.")
    parser.add_argument(
        "--simulate_user_intent", type=str, default="",
        help="Intent string for the simulated user. Defaults to the topic if not set.",
    )

    main(parser.parse_args())
