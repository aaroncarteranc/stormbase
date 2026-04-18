# KB Builder Design

**Date:** 2026-04-18  
**Status:** Approved  
**Scope:** Autonomous multi-article Obsidian knowledge base generation built on top of Co-STORM

---

## Overview

KB Builder is a new top-level module (`knowledge_storm/kb_builder/`) that wraps Co-STORM to autonomously generate a structured, multi-article Obsidian knowledge base from a single broad topic. It runs overnight on a 20-40B parameter model without user interaction.

The output is an Obsidian vault with three tiers of content:
1. **Original Documents** — full HTML→Markdown fetches of every source URL Co-STORM retrieves
2. **Indexed Documents** — AI-synthesized articles with wikilinks derived from original documents
3. **Discussion Documents** — cross-cutting thematic analyses synthesized after all indexed articles are complete

No existing Co-STORM code is modified. KB Builder is a thin orchestration layer above `CoStormRunner`.

---

## Architecture

### Module Structure

```
knowledge_storm/
  kb_builder/
    __init__.py
    orchestrator.py        # KBOrchestrator — main entry point
    article_queue.py       # ArticleQueue — priority queue, persisted to disk
    completion_checker.py  # CompletionChecker — hybrid per-article stopping logic
    expansion_planner.py   # ExpansionPlanner — gap analysis + curiosity expansion
    obsidian_exporter.py   # ObsidianExporter — formats output as Obsidian vault
    page_fetcher.py        # PageFetcher — HTTP fetch + HTML→Markdown conversion
```

### Component Responsibilities

**`KBOrchestrator`**
- Accepts: `topic`, `output_dir`, and run configuration parameters
- Seeds the `ArticleQueue` with 5–10 initial articles via an LLM call
- Drives the main loop: pop article → run Co-STORM session → export → expand
- Enforces hard ceilings and KB-level completion detection
- Triggers discussion document generation when the KB is complete

**`ArticleQueue`**
- Priority queue of pending articles (title, description, source, priority)
- Persisted to `<vault>/_meta/queue.json` after every mutation
- Enables crash recovery: a resumed run reads the queue and skips already-completed articles

**`CompletionChecker`**
- Implements the hybrid per-article stopping logic
- Uses `KnowledgeBaseSummaryModule` (already in Co-STORM) to get a compact KB tree summary
- Makes a single yes/no LLM call to assess sufficiency — reads only the node hierarchy, not article content

**`ExpansionPlanner`**
- Runs after each article completes
- Inputs: finished article title, current `kb-index.md`, inline curiosity candidates collected during the session
- Makes one LLM call proposing 0–3 new articles with title, description, and priority
- Deduplicates proposals against the queue and existing KB index (title similarity)
- Accumulates cross-cutting theme candidates throughout the run for later discussion generation

**`ObsidianExporter`**
- Converts `CoStormRunner.generate_report()` output to an indexed doc with frontmatter and wikilinks
- Replaces citation markers `[1]`, `[2]` with `[[original-documents/source-title]]` wikilinks
- Inserts wikilinks to other indexed articles where their titles appear in text
- Writes original documents via `PageFetcher`, deduplicating by URL across all articles
- Writes discussion documents after KB completion

**`PageFetcher`**
- Fetches source URLs using `httpx`
- Converts HTML to Markdown using `markdownify`
- Handles fetch failures gracefully (logs and falls back to the snippet Co-STORM already has)
- Caches fetched pages to disk to avoid redundant fetches across articles

---

## Vault Structure

```
<vault>/
  original-documents/
    <source-slug>.md        # one file per unique URL; full page as Markdown
  indexed/
    <article-slug>.md       # one file per KB article; AI-generated
  discussion/
    <theme-slug>.md         # one file per cross-cutting theme; AI-generated
  _meta/
    kb-index.md             # one-line summary per article; read by the curious LLM
    queue.json              # persisted article queue for crash recovery
    run-log.md              # human-readable progress log
```

### Frontmatter Schemas

**Original Document:**
```yaml
---
title: <page title>
url: <source url>
fetched_at: <ISO datetime>
cited_by:
  - "[[indexed/article-title]]"
---
```

**Indexed Document:**
```yaml
---
title: <article title>
topic: <broad topic>
created_at: <ISO datetime>
sources:
  - "[[original-documents/source-slug]]"
tags: []
---
```

**Discussion Document:**
```yaml
---
title: <theme title>
theme: <theme description>
created_at: <ISO datetime>
draws_from:
  - "[[indexed/article-title]]"
---
```

---

## Article Lifecycle

### Setup

A `CoStormRunner` is instantiated per article with:
- `topic` set to `"<broad topic> — <article title>"`
- `total_conv_turn` set to `max_ceiling` as a hard stop (overridden by `CompletionChecker`)

After `warm_start()`, the KB index is injected as the first user utterance via `runner.step(user_utterance=...)`. The utterance reads: "Here is what the knowledge base already covers — please avoid redundancy with these topics: <kb-index.md contents>". This uses `CoStormRunner`'s existing `step()` API without any modification.

### Hybrid Completion Loop

```
runner.warm_start()

for turn in range(max_ceiling):           # default: 40
    runner.step(simulate_user=True, ...)
    scan_utterance_for_curiosity_candidates(turn)

    if turn < min_floor:                  # default: 10
        continue
    if (turn - min_floor) % check_interval == 0:   # default: every 5 turns
        summary = KnowledgeBaseSummaryModule(runner.knowledge_base)
        if CompletionChecker.is_sufficient(summary, article_title):
            break

article_output = runner.generate_report()
```

### Inline Curiosity Flagging

After each conversation turn, expert utterances are scanned for entity/concept mentions not already present in the article queue or `kb-index.md`. This is a string match against known titles — no LLM call. Candidates are accumulated and passed to `ExpansionPlanner` after the article finishes.

---

## KB Expansion

`ExpansionPlanner` runs once per completed article with a single LLM call:

**Inputs:**
- Finished article title and one-line summary
- Full `kb-index.md` (titles + one-liners only)
- Inline curiosity candidates from the article's conversation

**Output:** 0–3 new article proposals (title, description, priority: high/medium/low)

Proposals are deduplicated against the queue and existing articles before being enqueued. Deduplication uses normalized string matching: lowercase, strip punctuation, strip common English stop words. An exact match on the normalized title is treated as a duplicate.

---

## KB Completion Detection

After each `ExpansionPlanner` run, the orchestrator checks two conditions (both must be true to stop):

1. **Queue silence** — `ExpansionPlanner` returned 0 new articles for the last `completion_silence_n` completed articles (default: 3)
2. **Semantic check** — a single LLM call reads `kb-index.md` and answers yes/no: "Does this KB give a thorough, well-rounded understanding of `<broad topic>`?"

If both pass, the run exits the main loop and proceeds to discussion generation.

---

## Hard Ceilings

All configurable via `KBOrchestrator` constructor parameters:

| Parameter | Default | Purpose |
|---|---|---|
| `max_articles` | 50 | Max total articles (completed + pending). Proposals exceeding this are dropped. |
| `max_expansion_rounds` | 10 | Max post-article gap checks before forcing KB completion regardless of semantic check. |
| `max_ceiling` (per article) | 40 | Max conversation turns per article session. |
| `min_floor` (per article) | 10 | Min conversation turns before completion checks begin. |
| `check_interval` (per article) | 5 | Turns between completion checks after floor. |

---

## Discussion Generation

After KB completion, `ExpansionPlanner`'s accumulated theme candidates are reviewed. Each theme becomes a discussion document:

- **Input to LLM:** theme description + wikilinks to relevant indexed docs (titles only, not content)
- **Output:** synthetic analysis in Markdown with `[[wikilinks]]` to source indexed docs
- Discussion docs do not re-read full indexed doc content — they reason from titles and themes

---

## Key Design Decisions

- **No Co-STORM modifications.** All new code lives in `kb_builder/`. Upstream `CoStormRunner` is used as-is.
- **`kb-index.md` as the LLM's window into the KB.** The curious LLM never reads full articles — only one-liners. This keeps token usage bounded on a 20-40B model.
- **Crash recovery via persisted queue.** Long overnight runs can be interrupted and resumed by re-reading `queue.json` and skipping completed articles.
- **URL deduplication for original docs.** A URL fetched for article A is reused for article B without a second HTTP request.
- **`markdownify` + `httpx` for page fetching.** Falls back to Co-STORM's existing snippet if a page fetch fails (paywalled, JS-heavy, etc.).
