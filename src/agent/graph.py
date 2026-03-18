"""Define the ReAct agent graph.

Works with a chat model with tool calling support.
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Literal, cast

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from agent.context import Context
from agent.dag import (
    append_node,
    build_dag_context,
    generate_id,
    init_jsonl,
    load,
    write_session,
)
from agent.state import InputState, State
from agent.tools import TOOLS
from agent.utils import get_message_text, load_chat_model

JSONL_DIR = Path("jsonls")
LOG_DIR = Path("logs")

# In-process cache: avoids re-loading DAG on every ReAct re-entry.
# Key = thread_id, value = Graph instance.
# Safe because a single invoke processes one thread at a time.
from agent.dag import Graph as _Graph

_dag_cache: dict[str, _Graph] = {}


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


async def _write_tool_log(thread_id: str, node_id: str, tools: list[dict]) -> int:
    """Append one tool-call log entry. Returns 1-based line number."""
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
            "ts": datetime.now(tz=UTC).isoformat(),
            "tools": tools,
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
    thread_id: str = config["configurable"]["thread_id"]  # type: ignore[index]
    if thread_id not in _dag_cache:
        jsonl_path = JSONL_DIR / f"{thread_id}.jsonl"
        await asyncio.to_thread(JSONL_DIR.mkdir, parents=True, exist_ok=True)
        if not await asyncio.to_thread(jsonl_path.exists):
            await asyncio.to_thread(init_jsonl, jsonl_path, graph_id=thread_id)
        _dag_cache[thread_id] = await asyncio.to_thread(load, jsonl_path)
    dag_graph = _dag_cache[thread_id]

    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)

    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

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
    thread_id: str = config["configurable"]["thread_id"]  # type: ignore[index]
    dag_graph = _dag_cache[thread_id]  # always populated by call_model
    jsonl_path = JSONL_DIR / f"{thread_id}.jsonl"

    # Extract tool calls with success/fail from ToolMessages
    tool_msg_map: dict[str, ToolMessage] = {
        m.tool_call_id: m
        for m in state.messages
        if isinstance(m, ToolMessage)
    }
    tools_used = []
    for msg in state.messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tm = tool_msg_map.get(tc["id"])
                ok = (getattr(tm, "status", "success") != "error") if tm else True
                tools_used.append({"name": tc["name"], "input": tc["args"], "ok": ok})

    # Pre-generate ULID so log entry and DAG node share the same ID
    node_id = generate_id()
    log_ref = await _write_tool_log(thread_id, node_id, tools_used) if tools_used else None

    # Generate summary
    model = load_chat_model(runtime.context.model).with_structured_output(QASummary)
    summary_prompt = (
        "Summarize the following Q-A exchange in ONE concise sentence. "
        "Capture the key question and the essence of the answer.\n\n"
        f"Question: {user_query}\n\n"
        f"Answer: {final_answer}"
    )
    try:
        result = cast(QASummary, await model.ainvoke([{"role": "user", "content": summary_prompt}]))
        summary_text = result.summary
    except Exception:
        summary_text = user_query[:100] + ("..." if len(user_query) > 100 else "")

    # Persist to DAG
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
    await asyncio.to_thread(write_session, dag_graph, jsonl_path)

    return {"summary": summary_text}


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

graph = builder.compile(name="ReAct Agent")
