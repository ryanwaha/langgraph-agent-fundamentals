"""Define the ReAct agent graph.

Works with a chat model with tool calling support.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Literal, cast

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from agent.context import Context
from agent.dag import (
    append_node,
    build_dag_context,
    build_merge_context,
    generate_id,
    init_jsonl,
    load,
    write_session,
)
from agent.state import InputState, State
from agent.tools import TOOLS
from agent.utils import get_message_text, load_chat_model

logger = logging.getLogger(__name__)

JSONL_DIR = Path("jsonls")
LOG_DIR = Path("logs")

# In-process cache: avoids re-loading DAG on every ReAct re-entry.
# Key = thread_id, value = Graph instance.
# Safe because a single invoke processes one thread at a time.
from agent.dag import Graph as _Graph

_dag_cache: dict[str, _Graph] = {}

# Per-invoke start time for duration tracking.
# Key = invoke_id (LangGraph thread_id / ULID), set on first call_model entry.
_turn_start: dict[str, datetime] = {}

# Session lifecycle: last activity timestamp per dag_thread_id.
# A session is flushed (write_session + evict cache) after SESSION_TIMEOUT_S of inactivity.
_last_activity: dict[str, datetime] = {}
SESSION_TIMEOUT_S = 600  # 10 minutes


# ---------------------------------------------------------------------------
# Structured output schema for the summarize node
# ---------------------------------------------------------------------------


class QASummary(BaseModel):
    """Structured output for Q-A summarization."""

    summary: str = Field(
        description="A single concise sentence summarizing the Q-A exchange."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _write_turn_log(
    thread_id: str,
    node_id: str,
    *,
    ts_start: datetime,
    ts_end: datetime,
    model: str,
    q: str,
    summary: str,
    steps: list[dict],
) -> int:
    """Append one complete turn log entry. Returns 1-based line number."""
    log_path = LOG_DIR / f"{thread_id}.jsonl"

    def _sync_write() -> int:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                line_num = sum(1 for _ in f) + 1
        else:
            line_num = 1
        entry = {
            "node_id": node_id,
            "ts_start": ts_start.isoformat(),
            "ts_end": ts_end.isoformat(),
            "duration_s": round((ts_end - ts_start).total_seconds(), 2),
            "model": model,
            "q": q,
            "summary": summary,
            "steps": steps,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return line_num

    return await asyncio.to_thread(_sync_write)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def call_model(
    state: State, runtime: Runtime[Context], config: RunnableConfig
) -> dict:
    """Call the LLM powering the agent."""
    # Load DAG on first entry; reuse from cache on ReAct re-entries (after tools)
    dag_thread_id: str = config["configurable"]["dag_thread_id"]  # type: ignore[index]
    invoke_id: str = config["configurable"]["thread_id"]  # type: ignore[index]
    if invoke_id not in _turn_start:
        _turn_start[invoke_id] = datetime.now(tz=UTC)
    if dag_thread_id not in _dag_cache:
        jsonl_path = JSONL_DIR / f"{dag_thread_id}.jsonl"
        await asyncio.to_thread(JSONL_DIR.mkdir, parents=True, exist_ok=True)
        if not await asyncio.to_thread(jsonl_path.exists):
            await asyncio.to_thread(init_jsonl, jsonl_path, graph_id=dag_thread_id)
        _dag_cache[dag_thread_id] = await asyncio.to_thread(load, jsonl_path)
    dag_graph = _dag_cache[dag_thread_id]

    # Merge parents can be passed via configurable (set by telegram_bot on /merge)
    merge_parents_cfg = config["configurable"].get("merge_parents")  # type: ignore[index]
    if merge_parents_cfg:
        dag_graph._merge_parents = merge_parents_cfg  # type: ignore[attr-defined]

    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)

    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    merge_parents = getattr(dag_graph, "_merge_parents", None)
    if merge_parents:
        dag_context = build_merge_context(dag_graph, merge_parents)
    else:
        dag_context = build_dag_context(dag_graph)
    if dag_context:
        system_message = system_message + "\n\n" + dag_context

    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ],
        }

    return {"messages": [response]}


async def summarize(
    state: State, runtime: Runtime[Context], config: RunnableConfig
) -> Dict[str, str]:
    """Summarize the Q-A exchange and persist to DAG."""
    user_query = get_message_text(state.messages[0])
    final_answer = get_message_text(state.messages[-1])
    dag_thread_id: str = config["configurable"]["dag_thread_id"]  # type: ignore[index]
    invoke_id: str = config["configurable"]["thread_id"]  # type: ignore[index]
    dag_graph = _dag_cache[dag_thread_id]  # always populated by call_model
    jsonl_path = JSONL_DIR / f"{dag_thread_id}.jsonl"

    ts_end = datetime.now(tz=UTC)
    ts_start = _turn_start.pop(invoke_id, ts_end)

    # Build a lookup: tool_call_id → (name, args) from all AIMessages
    tc_meta: dict[str, tuple[str, dict]] = {}
    for msg in state.messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_meta[tc["id"]] = (tc["name"], tc["args"])

    # Reconstruct ordered ReAct step trace from message history
    steps: list[dict] = []
    llm_iter = 0
    for msg in state.messages:
        if isinstance(msg, AIMessage):
            llm_iter += 1
            step: dict = {"type": "llm", "iter": llm_iter}
            if msg.tool_calls:
                step["tools_requested"] = [tc["name"] for tc in msg.tool_calls]
            steps.append(step)
        elif isinstance(msg, ToolMessage):
            name, args = tc_meta.get(msg.tool_call_id, ("unknown", {}))
            ok = (getattr(msg, "status", "success") != "error")
            output = str(msg.content)
            steps.append({
                "type": "tool",
                "iter": llm_iter,
                "name": name,
                "input": args,
                "output_preview": output[:300] if len(output) > 300 else output,
                "ok": ok,
            })


    # Pre-generate ULID so log entry and DAG node share the same ID
    node_id = generate_id()

    # Generate summary
    summary_model = load_chat_model(runtime.context.model).with_structured_output(QASummary)
    summary_prompt = (
        "Summarize the following Q-A exchange in ONE concise sentence. "
        "Capture the key question and the essence of the answer.\n\n"
        f"Question: {user_query}\n\n"
        f"Answer: {final_answer}"
    )
    try:
        result = cast(QASummary, await summary_model.ainvoke([{"role": "user", "content": summary_prompt}]))
        summary_text = result.summary
    except Exception:
        logger.warning(
            "summarize: LLM call failed for thread %s, falling back to truncated query",
            dag_thread_id,
            exc_info=True,
        )
        summary_text = user_query[:100] + ("..." if len(user_query) > 100 else "")

    # Write complete turn log (always, not just when tools were used)
    log_ref = await _write_turn_log(
        dag_thread_id,
        node_id,
        ts_start=ts_start,
        ts_end=ts_end,
        model=runtime.context.model,
        q=user_query,
        summary=summary_text,
        steps=steps,
    )

    # Persist to DAG — use merge parents if set, then consume
    merge_parents = getattr(dag_graph, "_merge_parents", None)
    if merge_parents:
        parents = merge_parents
        del dag_graph._merge_parents  # type: ignore[attr-defined]
    else:
        parents = [dag_graph.active_node] if dag_graph.active_node else []
    new_node = await asyncio.to_thread(
        append_node,
        dag_graph,
        jsonl_path,
        q=user_query,
        a=final_answer,
        sum_text=summary_text,
        parents=parents,
        node_id=node_id,
        log_ref=log_ref,
    )
    dag_graph.active_node = new_node.id
    _last_activity[dag_thread_id] = datetime.now(tz=UTC)

    return {"summary": summary_text, "last_node_id": node_id}


# ---------------------------------------------------------------------------
# Session lifecycle helpers
# ---------------------------------------------------------------------------


async def flush_session(dag_thread_id: str) -> bool:
    """Write session record and evict cache for one thread. Returns True if flushed."""
    dag_graph = _dag_cache.get(dag_thread_id)
    if dag_graph is None:
        return False
    jsonl_path = JSONL_DIR / f"{dag_thread_id}.jsonl"
    await asyncio.to_thread(write_session, dag_graph, jsonl_path)
    _dag_cache.pop(dag_thread_id, None)
    _last_activity.pop(dag_thread_id, None)
    logger.info("Session flushed: thread=%s", dag_thread_id)
    return True


async def flush_all_sessions() -> None:
    """Flush all open sessions (call on shutdown)."""
    for tid in list(_dag_cache):
        await flush_session(tid)


async def _session_watchdog() -> None:
    """Background task: flush sessions idle for SESSION_TIMEOUT_S seconds."""
    while True:
        await asyncio.sleep(60)
        now = datetime.now(tz=UTC)
        expired = [
            tid for tid, last in list(_last_activity.items())
            if (now - last).total_seconds() >= SESSION_TIMEOUT_S
        ]
        for tid in expired:
            await flush_session(tid)
            logger.info("Session timed out: thread=%s", tid)


def start_session_flusher() -> None:
    """Schedule the session watchdog. Call once after the event loop is running."""
    asyncio.get_event_loop().create_task(_session_watchdog())


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node(call_model)  # type: ignore[arg-type]
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node(summarize)  # type: ignore[arg-type]

builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["summarize", "tools"]:
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    if not last_message.tool_calls:
        return "summarize"
    return "tools"


builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")
builder.add_edge("summarize", "__end__")

graph = builder.compile(name="ReAct Agent", checkpointer=InMemorySaver())
