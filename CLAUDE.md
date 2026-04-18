# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**STORM** (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking) is a Python library (`knowledge-storm` on PyPI) that generates Wikipedia-like articles using LLMs and Internet search. It has two modes:

- **STORM**: Fully automated article generation (research → outline → article → polish)
- **Co-STORM**: Human-AI collaborative knowledge curation with a shared mind map

Both are implemented using [dspy](https://github.com/stanfordnlp/dspy).

## Setup

```bash
conda create -n storm python=3.11
conda activate storm
pip install -r requirements.txt
```

API keys go in `secrets.toml` at the root (see README for format).

## Key Commands

**Format code** (required before committing — enforced by pre-commit hook and CI):
```bash
black knowledge_storm/
```

**Install pre-commit hook** (formats code automatically before each commit):
```bash
pip install pre-commit
pre-commit install
```

**Run STORM example:**
```bash
python examples/storm_examples/run_storm_wiki_gpt.py \
    --output-dir $OUTPUT_DIR \
    --retriever bing \
    --do-research \
    --do-generate-outline \
    --do-generate-article \
    --do-polish-article
```

**Run Co-STORM example:**
```bash
python examples/costorm_examples/run_costorm_gpt.py \
    --output-dir $OUTPUT_DIR \
    --retriever bing
```

## Architecture

### Package structure: `knowledge_storm/`

- **`interface.py`** — Abstract base classes for all pipeline components. Start here when understanding or extending the system. Key abstractions:
  - `Engine` — base runner that auto-decorates `run_*` methods with timing/cost logging
  - `LMConfigs` — base config class; attributes ending in `_lm` are auto-collected for usage stats
  - `Retriever` / `KnowledgeCurationModule` / `OutlineGenerationModule` / `ArticleGenerationModule` / `ArticlePolishingModule` — the four pipeline stages
  - `Agent` — base for Co-STORM LLM agents
  - `Information` / `InformationTable` / `Article` / `ArticleSectionNode` — core data structures

- **`lm.py`** — Language model wrappers (all via litellm). `LitellmModel` is the primary class.

- **`rm.py`** — Retrieval modules / search engine integrations (`YouRM`, `BingSearch`, `VectorRM`, `SerperRM`, `BraveRM`, `DuckDuckGoSearchRM`, `TavilySearchRM`, `GoogleSearch`, `AzureAISearch`, `SearXNG`). PRs for new RMs are actively welcomed.

- **`encoder.py`** — Embedding model support (also via litellm).

- **`dataclass.py`** — Shared data structures for Co-STORM (`KnowledgeBase`, `ConversationTurn`, etc.).

- **`storm_wiki/`** — STORM implementation:
  - `engine.py` — `STORMWikiRunner` and `STORMWikiRunnerArguments`
  - `modules/` — concrete implementations of the four pipeline stages plus `persona_generator.py`, `storm_dataclass.py`

- **`collaborative_storm/`** — Co-STORM implementation:
  - `engine.py` — `CoStormRunner`, `RunnerArgument`, `CollaborativeStormLMConfigs`, `DiscourseManager`
  - `modules/` — agent types (`co_storm_agents.py`), warm-start logic, question generation, knowledge base summary, etc.

- **`utils.py`** — Shared utilities including `ArticleTextProcessing`.

- **`logging_wrapper.py`** — `LoggingWrapper` for tracking LM call history in Co-STORM.

### Multi-LM pattern

Both STORM and Co-STORM use a `LMConfigs` subclass where different pipeline stages are powered by different models. LM attributes must end in `_lm`; the base class auto-discovers them for usage tracking. A cheaper/faster model is recommended for high-frequency tasks (conversation simulation, query splitting), and a stronger model for article generation.

### Versioning

Version must be kept in sync between `setup.py` and `knowledge_storm/__init__.py` — CI enforces this on package builds.

## Contribution Scope

The project currently accepts PRs for:
- New LM support in `knowledge_storm/lm.py`
- New search engine/retriever support in `knowledge_storm/rm.py`
- New features for `frontend/demo_light/`
- Bug reports and issue responses

Code refactoring PRs are not accepted to avoid conflicts with the core team's work.

## Code Style

- `black` formatter, applied only to `knowledge_storm/` directory
- PR titles should follow: `[New LM/New RM/Demo Enhancement] <description>`
