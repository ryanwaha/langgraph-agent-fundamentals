# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Status (2026-03-19)

| Component | Status | Notes |
|-----------|--------|-------|
| Telegram bot | ✅ Active | Streaming CoT, sendMessageDraft, markdown-it-py, inline buttons, debouncing, per-thread lock |
| Discord bot | 🟡 Disabled | `DISCORD_BOT_TOKEN=null` in `.env`; no DAG integration |
| LangGraph Studio | ✅ Ready | `langgraph dev` works |
| jsonl-dag-engine | ✅ Populated | git submodule at `src/jsonl-dag-engine/`, branch `feat/log-ref-node-id` (commit `6e8b14b`) |
| Checkpointer | 🟡 Not added | `graph.compile()` has no checkpointer; Discord thread memory is not persisted |

## Environment

- **Conda env**: `LG` — always use `conda activate LG` before running anything
- **Python**: `C:/Users/ryanw/miniconda3/envs/LG/python.exe`
- **Package layout**: `src/agent/` is the importable `agent` package (configured in `pyproject.toml` + `pyrightconfig.json`)

## Running the project

```bash
# Start LangGraph Studio (test agent without Discord)
langgraph dev --no-browser
# Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

# Start Discord bot
PYTHONPATH=src python -m bot.discord_bot

# Start Telegram bot
PYTHONPATH=src python -m bot.telegram_bot
```

> **Windows encoding**: `.env` must use ASCII-only comments (no em dashes `—`). The `langgraph dev` dotenv parser uses the system codepage (cp950 on this machine).

## Key dependencies

| Package | Purpose |
|---------|---------|
| `langgraph>=1.0.9` | Core graph runtime |
| `langchain-google-genai>=4.2.1` | Gemini model support |
| `langchain-tavily>=0.2.17` | Web search tool |
| `discord.py>=2.7.1` | Discord gateway client |
| `python-telegram-bot>=20.0` | Telegram async bot client (installed: 22.6, uses `sendMessageDraft`) |
| `markdown-it-py>=3.0.0` | CommonMark parser for `_md_to_html()` in telegram bot |
| `python-ulid>=3.0.0` | ULID generation for DAG node IDs |
| `langgraph-cli[inmem]` | `langgraph dev` Studio server |

Default model: `google_genai/gemini-2.5-flash-lite` (set via `MODEL` in `.env`).

## Telegram bot commands

`/list [n]`, `/branch <prefix>`, `/switch <prefix>`, `/status`, `/show`, `/paths`, `/delete <prefix>`, `/render`, `/maintain`

## Design decisions — do not change without reason

**`_dag_cache`** (`graph.py`): module-level `dict[str, Graph]`, keyed by `thread_id`. Intentionally kept outside LangGraph `State` to avoid serialization overhead. Safe for single-process; not safe for multi-worker. Do not move into State unless switching to multi-worker deployment.

**`# type: ignore` in `telegram_bot.py`**:
- `bot.send_message_draft(...)` and `bot.send_message(...)` — `bot` is typed as `object` intentionally to avoid PTB `ExtBot` import complexity. The ignores are correct; do not remove.

**`# type: ignore` in `graph.py`**:
- `config["configurable"]["thread_id"]` — `RunnableConfig.configurable` is `Optional[Dict]`; always populated by the bot at runtime
- `cast(QASummary, result).summary` — `with_structured_output` return type is ambiguous to Pylance
- `builder.add_node(call_model)` — LangGraph's `(state, runtime, config)` node signature is not in Pylance's stubs

## Type checking (Pylance)

`pyrightconfig.json` sets `extraPaths: ["src", "src/jsonl-dag-engine"]`. Both paths are required — removing either causes ~18 cascading errors across `dag.py`, `graph.py`, `state.py`, and `telegram_bot.py`.
