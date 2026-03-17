# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Status (2026-03-17)

| Component | Status | Notes |
|-----------|--------|-------|
| Telegram bot | ✅ Active | DAG lifecycle inside agent; bot layer is a thin wrapper |
| Discord bot | 🟡 Disabled | `DISCORD_BOT_TOKEN=null` in `.env`; no DAG integration |
| LangGraph Studio | ✅ Ready | `langgraph dev` works |
| jsonl-dag-engine | ✅ Populated | git submodule at `src/jsonl-dag-engine/`, branch `feat/log-ref-node-id` (commit `6e8b14b`) |
| Checkpointer | 🟡 Not added | `graph.compile()` has no checkpointer; Discord thread memory is not persisted |

**Known gaps**:
- Discord bot lacks DAG conversation memory (Telegram is the only frontend with full history)
- No LangGraph checkpointer — if needed, add to `graph.compile()` in `graph.py`
- `plan_draft.md` at project root tracks ongoing design plans

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

## Architecture

### Two-layer design

```
Discord  (bot/discord_bot.py)  ─┐
                                 ├─ graph.ainvoke(InputState(...), config=RunnableConfig(...), context=Context())
Telegram (bot/telegram_bot.py) ─┘
                                 ↓
                    LangGraph ReAct Agent (src/agent/)
                         call_model → tools → summarize
                         (DAG load / context inject / persist all happen here)
```

The bot layers and the agent are fully decoupled — `graph.ainvoke()` can be called from any frontend.

### Agent internals (`src/agent/`)

| File | Role |
|------|------|
| `graph.py` | StateGraph: `call_model` ↔ `tools` loop → `summarize` node; owns the full DAG lifecycle |
| `state.py` | `InputState` (messages only) and `State` (+ `is_last_step`, `summary`, `dag_graph`) |
| `context.py` | `Context` dataclass — model, system prompt, max_search_results. `__post_init__` auto-reads env vars. |
| `dag.py` | sys.path shim for `src/jsonl-dag-engine/` + `build_dag_context()` helper |
| `tools.py` | `TOOLS` list + `@register_tool` decorator — add a new tool by decorating an async function |
| `prompts.py` | `SYSTEM_PROMPT` template (supports `{system_time}`) |
| `utils.py` | `load_chat_model("provider/model")` and `get_message_text()` |

### Context injection

`Context` is passed as `context=Context()` to `graph.ainvoke()` — **not** via `configurable`. This is required for standalone (non-LangGraph-Platform) operation. The LangGraph `Runtime[Context]` object inside nodes is populated this way.

### Tool registry

```python
@register_tool
async def my_tool(param: str) -> dict:
    """Tool description shown to the LLM."""
    runtime = get_runtime(Context)          # access context.max_search_results etc.
    ...
```

Decorating with `@register_tool` is all that's needed — the tool is automatically included in `TOOLS` and bound to the model.

### Discord thread isolation

Each Discord channel gets its own `thread_id = str(message.channel.id)`. Enabling per-channel conversation memory only requires adding a checkpointer to `graph.compile()` in `graph.py`.

### Telegram chat isolation

Each Telegram chat gets its own `thread_id = str(message.chat_id)`. Forum topics use `f"{chat_id}_{message_thread_id}"`. Conversation history is persisted via the DAG engine (see below).

### DAG conversation memory

JSONL-DAG-engine (`src/jsonl-dag-engine/`, git submodule) provides append-only DAG-structured conversation persistence. One `.jsonl` file per thread under `jsonls/`.

- `src/agent/dag.py` — import shim + `build_dag_context()`: renders prior conversation as XML injected into the system message
- `src/agent/graph.py` — owns the full DAG lifecycle (see below)
- Bot commands: `/list [n]`, `/branch <prefix>`, `/switch <prefix>` (alias)

## Key dependencies

| Package | Purpose |
|---------|---------|
| `langgraph>=1.0.9` | Core graph runtime |
| `langchain-google-genai>=4.2.1` | Gemini model support |
| `langchain-tavily>=0.2.17` | Web search tool |
| `discord.py>=2.7.1` | Discord gateway client |
| `python-telegram-bot>=20.0` | Telegram async bot client |
| `python-ulid>=3.0.0` | ULID generation for DAG node IDs |
| `langgraph-cli[inmem]` | `langgraph dev` Studio server |

Default model: `google_genai/gemini-2.5-flash-lite` (set via `MODEL` in `.env`).

## DAG engine integration notes

`src/agent/dag.py` is the bridge between the agent and jsonl-dag-engine:
- Adds `src/jsonl-dag-engine/` to `sys.path` at import time
- Re-exports all public DAG API (`load`, `append_node`, `write_session`, `switch_active`, `init_jsonl`, `generate_id`, `Graph`, `Node`, `PromptBuilder`)
- `build_dag_context(dag_graph)` renders prior conversation as `<conversation>` + `<narrative>` XML (via `PromptBuilder`), returned as a string appended to the system message in `graph.py`

**Agent DAG lifecycle** (inside `src/agent/graph.py` nodes):
1. `call_model` (first entry, `state.dag_graph is None`): `init_jsonl` / `load` → `build_dag_context` → append context to system message → return `dag_graph` in state
2. `call_model` (ReAct re-entries): reuses `state.dag_graph` directly — no disk I/O
3. `summarize`: extracts tool trace from `state.messages`, pre-generates ULID, writes `logs/{thread_id}.jsonl` (tool log), generates QA summary, calls `append_node` + `write_session`

**Tool log** (`logs/{thread_id}.jsonl`): one entry per Q-A turn that used tools. Each entry contains `node_id` (same ULID as the JDE node), `ts`, and `tools` list with `name`/`input`/`ok` per call. `node.log_ref` stores the 1-based line number in the log file.

**JSONL storage**: `jsonls/{thread_id}.jsonl` — one file per Telegram chat (real data exists at `jsonls/7127827719.jsonl`)

**Record types** in JSONL: `meta`, `node` (id/q/a/sum/parents/log_ref), `session` (active_node snapshot), `tombstone` (soft delete)

**JDE extensions** (branch `feat/log-ref-node-id`):
- `Node.log_ref: int | None` — nullable link to tool log line number
- `append_node(..., node_id=None, log_ref=None)` — caller can supply a pre-generated ULID and log reference

## Type checking (Pylance)

`pyrightconfig.json` sets `extraPaths: ["src", "src/jsonl-dag-engine"]`. Both paths are required:
- `"src"` — resolves the `agent` package
- `"src/jsonl-dag-engine"` — resolves `dag_engine` and `prompt_builder` (bare imports from the submodule)

Without `"src/jsonl-dag-engine"`, Pylance cannot resolve the submodule imports and cascades ~18 errors across `dag.py`, `graph.py`, `state.py`, and `telegram_bot.py`.

Remaining `# type: ignore` annotations in `graph.py`:
- `config["configurable"]["thread_id"]  # type: ignore[index]` — `RunnableConfig.configurable` is typed `Optional[Dict]`; we know it's always populated by the bot
- `cast(QASummary, result).summary` — `with_structured_output` return type is ambiguous to Pylance
- `builder.add_node(call_model)  # type: ignore[arg-type]` — LangGraph's `(state, runtime, config)` node signature is not in Pylance's stubs
