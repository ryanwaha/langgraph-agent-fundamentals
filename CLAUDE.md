# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
python -m bot.discord_bot

# Start Telegram bot
python -m bot.telegram_bot
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
```

The bot layers and the agent are fully decoupled — `graph.ainvoke()` can be called from any frontend.

### Agent internals (`src/agent/`)

| File | Role |
|------|------|
| `graph.py` | StateGraph with two nodes: `call_model` → `tools` loop |
| `state.py` | `InputState` (messages only) and `State` (+ `is_last_step`) |
| `context.py` | `Context` dataclass — model name, system prompt, max_search_results. `__post_init__` auto-reads env vars (`MODEL`, `MAX_SEARCH_RESULTS`). |
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

Each Telegram chat gets its own `thread_id = str(message.chat_id)`. Private chats use the user's chat ID; group chats use the group's chat ID. Enabling per-chat conversation memory requires the same checkpointer change in `graph.py`.

## Key dependencies

| Package | Purpose |
|---------|---------|
| `langgraph>=1.0.9` | Core graph runtime |
| `langchain-google-genai>=4.2.1` | Gemini model support |
| `langchain-tavily>=0.2.17` | Web search tool |
| `discord.py>=2.7.1` | Discord gateway client |
| `python-telegram-bot>=20.0` | Telegram async bot client |
| `langgraph-cli[inmem]` | `langgraph dev` Studio server |

Default model: `google_genai/gemini-2.5-flash-lite` (set via `MODEL` in `.env`).
