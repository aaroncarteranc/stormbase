# Co-STORM: Architecture, Capabilities, and Extension Guide

## What Co-STORM Does

Co-STORM is a human-AI collaborative knowledge curation system. Where STORM generates a Wikipedia-like article fully automatically, Co-STORM builds the same artifact through a live roundtable conversation between the user and a panel of simulated domain experts. The result is both the conversation transcript and a structured mind map (knowledge base) that can be exported as a report at any point.

The key artifact is a **shared knowledge base** — a tree of topic nodes, each holding a set of retrieved and cited document snippets. The conversation is the mechanism by which that tree gets populated.

---

## Architecture Overview

```
CoStormRunner
├── CollaborativeStormLMConfigs   (6 LM slots)
├── RunnerArgument                (all tuning knobs)
├── KnowledgeBase                 (the mind map / shared state)
├── DiscourseManager              (turn policy + agent registry)
│   ├── Moderator
│   ├── CoStormExpert × N         (rotating panel)
│   ├── SimulatedUser             (experiments only)
│   └── PureRAGAgent              (baseline mode only)
└── LoggingWrapper / CallbackHandler
```

### Entry Point

`CoStormRunner` is the only public API. The typical call sequence:

```python
runner = CoStormRunner(lm_config, runner_argument, logging_wrapper, rm=search_backend)
runner.warm_start()               # bootstraps KB before first user message
runner.step(user_utterance=None)  # system generates next turn
runner.step(user_utterance="...")  # user injects a message
report = runner.generate_report() # exports KB as article text
```

---

## The Six LM Slots

Different pipeline stages use different models. All six slots must be set to a `LitellmModel` instance. Cheaper/faster models are appropriate for high-frequency low-stakes calls; stronger models for generation.

| Slot | Default | Primary Callers | Output Size |
|---|---|---|---|
| `question_answering_lm` | gpt-4o | `QuestionToQuery`, `AnswerQuestion`, `SimulatedUser` | 1000 tok |
| `discourse_manage_lm` | gpt-4o | `GenExpertActionPlanning`, `GenerateExpertModule` | 500 tok |
| `utterance_polishing_lm` | gpt-4o | `ConvertUtteranceStyle` (expert polish pass) | 2000 tok |
| `warmstart_outline_gen_lm` | gpt-4-1106-preview | `GenerateWarmStartOutlineModule` | 500 tok |
| `question_asking_lm` | gpt-4o | `WarmStartModerator`, `GroundedQuestionGenerationModule` | 300 tok |
| `knowledge_base_lm` | gpt-4o | `InsertInformationModule`, `ExpandNodeModule`, `ArticleGenerationModule`, `KnowledgeBaseSummaryModule` | 1000 tok |

The LM attribute naming convention (`_lm` suffix) is load-bearing — the base class auto-discovers these for usage tracking.

---

## Agents

### CoStormExpert
The primary answering agent. Each expert has a `persona` string (e.g., "Computational Linguist focused on grounding") and generates grounded answers by:
1. Fetching the current KB summary
2. Planning an action type: `[Original Question | Further Details | Information Request | Potential Answer]`
3. If answering: generating search queries → retrieving → synthesizing a cited answer
4. Polishing the utterance into conversational style

Expert personas are generated fresh after each user question by `GenerateExpertModule`. The panel is a rotating queue of `max_num_round_table_experts` (default 2) active experts.

**LM calls per expert turn:** ~4–5 LM calls + 1–2 search calls.

### Moderator
Fires every `moderator_override_N_consecutive_answering_turn` (default 3) answer turns. Does **not** call retrieval — it mines already-retrieved-but-uncited snippets from recent turns and asks a pointed follow-up question to steer conversation into underexplored territory.

Snippet scoring: `score = (1 - query_sim)^0.5 × (1 - cited_sim)^0.5 × claim_sim_mask`
Favors snippets that are on-topic but not yet incorporated into the KB.

### SimulatedUser
Plays the human role for automated experiments. Requires an `intent` string. Uses the same `AskQuestionWithPersona` module from STORM.

### PureRAGAgent
Baseline mode only. No action planning, no KB summary, no style polish. Directly answers the last question via `AnswerQuestionModule`.

---

## Conversation Turn Flow

```
User injects message
└─→ Append ConversationTurn(role="Guest") — no LM calls

System generates turn:
├── DiscourseManager.get_next_turn_policy() → TurnPolicySpec
│     [moderator override?] → Moderator
│     [last turn was Q?]    → general_knowledge_provider + update_experts_list=True + polish=True
│     [else]                → rotate expert queue + polish=True
│     [every N turns]       → Moderator + reorganize_kb=True
│
├── agent.generate_utterance(knowledge_base, conversation_history)
│     [CoStormExpert path]
│     ├── KB summary (knowledge_base_lm)
│     ├── GenExpertActionPlanning (discourse_manage_lm)
│     ├── QuestionToQuery (question_answering_lm)
│     ├── retriever.retrieve() (parallel search)
│     ├── AnswerQuestion (question_answering_lm)
│     └── ConvertUtteranceStyle (utterance_polishing_lm)
│
├── [if update_experts_list] GenerateExpertModule (discourse_manage_lm)
│
├── knowledge_base.update_from_conv_turn()
│     └── InsertInformationModule (encoder + knowledge_base_lm per unique intent)
│
└── [if reorganize_kb] knowledge_base.reorganize()
      ├── trim/merge tree housekeeping
      └── ExpandNodeModule (knowledge_base_lm) for over-full nodes
```

---

## Warm Start

Warm start runs before the first user message to seed the KB with initial research. It is a compressed version of the STORM pipeline:

1. **Background retrieval** — `AnswerQuestionModule` in "extensive" mode on the bare topic
2. **Expert generation** — `GenerateExpertModule` produces `warmstart_max_num_experts` (default 3) personas
3. **Perspective-guided QA** — each expert asks `warmstart_max_turn_per_experts` (default 2) questions; answers are retrieved and synthesized. Runs in parallel across experts.
4. **Outline generation** — draft outline → refined outline → `knowledge_base.insert_from_outline_string()` builds the initial tree structure
5. **KB population** — warm-start turns inserted into the new tree via `InsertInformationModule`
6. **Intro conversion** — `ReportToConversation` converts each KB node into a Q+A exchange, producing the engaging warm-start conversation shown to the user

**LM call count at default settings:** ~30–60 calls before the first user message.
**After warm start:** `discourse_manager.next_turn_moderator_override = True`, so the first system turn is always the Moderator.

---

## Knowledge Base

The KB is a **tree of topic nodes** (`KnowledgeNode`), each holding a set of citation UUIDs. The global registry maps UUID → `Information` (a retrieved document with url, title, snippets, and metadata). Deduplication is hash-based.

### Insertion

`InsertInformationModule` places a cited snippet into the tree:
1. **Embedding-based candidate selection**: encode the (query, question) intent, compare against cached KB structure embedding, present top-5 nodes to `InsertInformationCandidateChoice` LLM
2. **Fallback — layer-by-layer navigation**: iteratively call `InsertInformation` (ChainOfThought) at each depth level; LLM chooses `insert` / `step: [child]` / `create: [new_child]`

### Expansion

When a node accumulates ≥ `node_expansion_trigger_count` (default 10) citations, `ExpandNodeModule` fires: an LLM names subsections, the node's citations are re-inserted under the new children, and the original node is cleared.

### Report Generation

`knowledge_base.to_report()` calls `ArticleGenerationModule` which runs `WriteSection` in parallel (5 threads) over all non-empty nodes, synthesizing each into a Wikipedia-style paragraph. Results are cached in `node.synthesize_output` with a dirty-flag (`need_regenerate_synthesize_output`) to avoid re-generating unchanged nodes.

---

## Tuning Knobs

Key `RunnerArgument` fields for controlling behavior:

| Field | Default | Effect |
|---|---|---|
| `retrieve_top_k` | 10 | Documents per search query |
| `max_search_queries` | 2 | Queries generated per QA call |
| `warmstart_max_num_experts` | 3 | Expert perspectives in warm start |
| `warmstart_max_turn_per_experts` | 2 | Q&A rounds per warm-start expert |
| `max_num_round_table_experts` | 2 | Active experts in rotation |
| `moderator_override_N_consecutive_answering_turn` | 3 | Moderator frequency |
| `node_expansion_trigger_count` | 10 | Snippets before node splits |
| `disable_moderator` | False | Remove Moderator from policy |
| `disable_multi_experts` | False | Single-expert mode |
| `rag_only_baseline_mode` | False | Skip all Co-STORM logic |

---

## Known Fragility Points

These are brittle for any model and particularly problematic for smaller models:

### 1. Structured Output Parsing
`GenExpertActionPlanning` must output exactly `[Original Question]: ...` / `[Further Details]: ...` etc. The parser checks for these literal prefixes. Any preamble, paraphrase, or alternative format raises an uncaught `Exception` with no retry (`costorm_expert_utterance_generator.py:139`).

### 2. KB Navigation Naming
Layer-by-layer navigation (`InsertInformation`) requires the model to output exact child node names. Paraphrased or slightly different names cause a `ValueError: Child node with name X not found`. The embedding-based first-pass exists specifically to reduce how often this fallback runs, but it still fires for novel intents.

### 3. KB Summary Cost Growth
`knowledge_base.get_knowledge_base_summary()` is called on every expert turn. As the KB grows, the full hierarchy string (passed to `KnowledgeBaseSummaryModule`) grows linearly with the number of nodes. Late in a long conversation this can consume a significant fraction of the model's context window.

### 4. `ReportToConversation` Format Dependency
The warm-start Q+A extraction splits on literal `"Question:"` / `"Answer:"` prefixes. Models that omit these headers produce blank warm-start conversation entries silently.

### 5. `from_dict()` LM Config Bug
Session deserialization (`engine.py:556-557`) always re-initializes `lm_config` from environment variables, discarding the serialized config. Sessions resumed via `from_dict()` revert to the default OpenAI/Azure preset.

### 6. Moderator Embedding Cost
Every Moderator turn batch-encodes all snippets from the last 2 answer turns. With default settings this can be 20+ documents × multiple snippets each. There is no per-session embedding cache for raw retrieved snippets (only KB structure is cached).

---

## Extension Avenues

### Small Model Compatibility (20–40B Range)

The core challenge is that Co-STORM was designed around GPT-4-class instruction following. Smaller models struggle with:

**Output format compliance** — The most impactful single change is making all structured outputs more robust:
- Replace the `[Type]: content` action planning format with a JSON field or dspy `Literal` field constraint; add a retry-with-feedback loop on parse failure
- Replace layer-by-layer tree navigation with a single call that sees the full tree and returns a path; reduces both call count and name-matching failures
- Add a `max_retries=2` wrapper around `GenExpertActionPlanning` before raising

**Context window management** — KB summary on every turn is expensive:
- Cache `get_knowledge_base_summary()` output and invalidate only when `need_regenerate_synthesize_output` is set on any node (the dirty flag already exists)
- Offer a "compact KB summary" mode that shows only the top two tree levels rather than the full hierarchy

**LM call reduction** — Warm start is the biggest per-session cost:
- `warmstart_max_num_experts=1` + `warmstart_max_turn_per_experts=1` reduces warm-start LM calls from ~50 to ~10 with minimal quality loss for smaller topics
- Skip `ReportToConversation` and surface the raw KB outline instead; saves article-generation LM calls during warm start

**Simplified action taxonomy** — Reduce the 4-way action classification to binary: `[Question]` vs `[Answer]`. Smaller models handle binary decisions more reliably.

### New Agent Types

The `Agent` base class (`interface.py`) is minimal — implement `generate_utterance(knowledge_base, conversation_history) -> ConversationTurn` and register the instance in `DiscourseManager`. Promising additions:

- **Skeptic agent**: reviews existing KB claims against new retrieved evidence, generates contradicting or qualifying utterances
- **Synthesizer agent**: fires periodically to propose merges across KB nodes, rather than only expanding
- **Citation auditor**: validates that cited snippets actually support the claim in the utterance, flags unsupported assertions

### Retrieval Improvements

`AnswerQuestionModule` uses a single-stage retrieve → answer pipeline. Extensions:
- **Iterative retrieval**: let the LLM decide whether retrieved results are sufficient before answering; loop if not (2–3 iterations max)
- **Diversity-aware retrieval**: de-duplicate retrieved snippets at the retriever level by semantic similarity before passing to the answer module, reducing redundancy in the KB
- **Dense + sparse hybrid**: `VectorRM` already exists for dense retrieval; a hybrid re-ranker over dense + BM25 would improve recall for technical topics

### Knowledge Base Extensions

- **Conflict tracking**: add a `conflicts: Set[int]` field to `KnowledgeNode` alongside `content`; populate it when `InsertInformationModule` detects contradictory snippets at the same node
- **Confidence scoring**: weight citations by source quality metadata (if the retriever provides it) and surface low-confidence nodes in the report
- **Cross-references**: allow `KnowledgeNode` to hold soft links to sibling nodes rather than only parent–child relationships; needed when topics genuinely overlap

### Discourse Policy Extensions

The `get_next_turn_policy()` method in `DiscourseManager` (`engine.py:461-502`) is the single choke point for all turn-taking logic. It is a simple if/elif chain and is easy to extend:

- **User intent detection**: classify the user utterance type before deciding policy (question vs. correction vs. elaboration request) and route differently
- **Adaptive moderator frequency**: count how many new snippets were inserted in the last N turns; if the KB stopped growing, trigger Moderator earlier
- **Expert specialization gate**: only route a question to an expert if their persona keyword-matches the question topic, rather than strict round-robin

### Callback / Observability

`BaseCallbackHandler` (`callback.py`) has hooks at every major pipeline stage. The existing `LocalConsolePrintCallBackHandler` only prints. A richer implementation could:
- Emit structured events to a message queue for real-time UI updates
- Track per-turn LM cost and surface it alongside each utterance
- Record which KB nodes were modified per turn for diff visualization

---

## Files Quick Reference

| File | Role |
|---|---|
| [engine.py](../knowledge_storm/collaborative_storm/engine.py) | Runner, configs, DiscourseManager |
| [co_storm_agents.py](../knowledge_storm/collaborative_storm/modules/co_storm_agents.py) | All 4 agent classes |
| [costorm_expert_utterance_generator.py](../knowledge_storm/collaborative_storm/modules/costorm_expert_utterance_generator.py) | Expert turn 3-step pipeline |
| [grounded_question_answering.py](../knowledge_storm/collaborative_storm/modules/grounded_question_answering.py) | Core RAG module |
| [grounded_question_generation.py](../knowledge_storm/collaborative_storm/modules/grounded_question_generation.py) | Moderator question gen + style polish |
| [information_insertion_module.py](../knowledge_storm/collaborative_storm/modules/information_insertion_module.py) | KB insertion and expansion |
| [warmstart_hierarchical_chat.py](../knowledge_storm/collaborative_storm/modules/warmstart_hierarchical_chat.py) | Full warm-start pipeline |
| [article_generation.py](../knowledge_storm/collaborative_storm/modules/article_generation.py) | Report/article synthesis |
| [expert_generation.py](../knowledge_storm/collaborative_storm/modules/expert_generation.py) | Expert persona generation |
| [knowledge_base_summary.py](../knowledge_storm/collaborative_storm/modules/knowledge_base_summary.py) | KB → summary text |
| [dataclass.py](../knowledge_storm/dataclass.py) | KnowledgeBase, KnowledgeNode, ConversationTurn |
| [interface.py](../knowledge_storm/interface.py) | Information, Retriever, Agent base classes |
| [collaborative_storm_utils.py](../knowledge_storm/collaborative_storm/modules/collaborative_storm_utils.py) | Formatting utilities, module factory |
| [callback.py](../knowledge_storm/collaborative_storm/modules/callback.py) | Callback interface and console impl |
| [encoder.py](../knowledge_storm/encoder.py) | Embedding engine (text-embedding-3-small via litellm) |
