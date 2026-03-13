"""Define the ReAct agent graph.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from agent.context import Context
from agent.state import InputState, State
from agent.tools import TOOLS
from agent.utils import get_message_text, load_chat_model


# ---------------------------------------------------------------------------
# Structured output schema for the summarize node
# ---------------------------------------------------------------------------


class QASummary(BaseModel):
    """Structured output for Q-A summarization."""

    summary: str = Field(
        description="A single concise sentence summarizing the Q-A exchange."
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering the agent.

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state: The current state of the conversation.
        runtime: Runtime context containing model and configuration.

    Returns:
        A dictionary containing the model's response message.
    """
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)

    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Append DAG conversation context if available
    if runtime.context.dag_context:
        system_message = system_message + "\n\n" + runtime.context.dag_context

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
            ]
        }

    return {"messages": [response]}


async def summarize(
    state: State, runtime: Runtime[Context]
) -> Dict[str, str]:
    """Generate a one-sentence summary of the completed Q-A exchange.

    Called after the ReAct loop ends (no more tool calls).  Extracts the
    user's question and the agent's final answer, then uses structured
    output to produce a dense summary for DAG persistence.
    """
    user_query = get_message_text(state.messages[0])
    final_answer = get_message_text(state.messages[-1])

    model = load_chat_model(runtime.context.model).with_structured_output(QASummary)

    summary_prompt = (
        "Summarize the following Q-A exchange in ONE concise sentence. "
        "Capture the key question and the essence of the answer.\n\n"
        f"Question: {user_query}\n\n"
        f"Answer: {final_answer}"
    )

    try:
        result = await model.ainvoke(
            [{"role": "user", "content": summary_prompt}]
        )
        return {"summary": result.summary}
    except Exception:
        # Fallback: truncated user query as summary
        fallback = user_query[:100] + ("..." if len(user_query) > 100 else "")
        return {"summary": fallback}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node(call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node(summarize)

builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["summarize", "tools"]:
    """Determine the next node based on the model's output.

    Args:
        state: The current state of the conversation.

    Returns:
        The name of the next node to call ("summarize" or "tools").
    """
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
