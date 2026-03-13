"""Telegram bot entry point.

Bridges Telegram messages to the LangGraph ReAct agent with DAG-based
conversation persistence.  Each chat (or forum topic) gets its own JSONL
file, enabling branching conversation history.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.context import Context
from agent.dag import (
    append_node,
    build_dag_context,
    init_jsonl,
    load,
    switch_active,
    write_session,
)
from agent.graph import graph
from agent.state import InputState
from agent.utils import get_message_text

load_dotenv()

logger = logging.getLogger(__name__)

JSONL_DIR = Path("jsonls")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _thread_id(message) -> str:
    """Derive a unique thread ID from a Telegram message.

    - Forum topic  -> ``{chat_id}_{message_thread_id}``
    - Regular chat -> ``{chat_id}``
    """
    if message.message_thread_id is not None:
        return f"{message.chat_id}_{message.message_thread_id}"
    return str(message.chat_id)


def _jsonl_path(thread_id: str) -> Path:
    return JSONL_DIR / f"{thread_id}.jsonl"


# ---------------------------------------------------------------------------
# Graph invocation
# ---------------------------------------------------------------------------


async def run_agent(user_message: str, thread_id: str) -> str:
    """Invoke the LangGraph agent with DAG context and persist the result.

    Flow:
    1. Load (or initialize) the JSONL file for this thread.
    2. Build DAG context from the conversation graph.
    3. Invoke the LangGraph agent with dag_context in Context.
    4. Extract the answer and summary from the result.
    5. Append a new node to the DAG.
    6. Write a session record to persist active_node.

    Args:
        user_message: The raw text from the Telegram user.
        thread_id: Unique identifier for this conversation thread.

    Returns:
        The agent's final text response.
    """
    jsonl_path = _jsonl_path(thread_id)

    # Load or initialize the DAG
    if jsonl_path.exists():
        dag_graph = load(jsonl_path)
    else:
        init_jsonl(jsonl_path, graph_id=thread_id)
        dag_graph = load(jsonl_path)

    # Build DAG context (empty string if no nodes yet)
    dag_context = build_dag_context(dag_graph)

    config = RunnableConfig(configurable={"thread_id": thread_id})

    result = await graph.ainvoke(
        InputState(messages=[HumanMessage(content=user_message)]),
        config=config,
        context=Context(dag_context=dag_context),
    )

    # Extract answer and summary
    final_message = result["messages"][-1]
    answer = get_message_text(final_message)
    summary = result.get("summary", "")

    # Determine parent node(s)
    parents = [dag_graph.active_node] if dag_graph.active_node else []

    # Append the new node to the DAG
    new_node = append_node(
        dag_graph,
        jsonl_path,
        q=user_message,
        a=answer,
        sum_text=summary,
        parents=parents,
    )

    # Update active node and persist
    dag_graph.active_node = new_node.id
    write_session(dag_graph, jsonl_path)

    return answer


# ---------------------------------------------------------------------------
# Telegram event handlers
# ---------------------------------------------------------------------------


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming Telegram messages.

    In private chats, always responds.
    In group chats, only responds when @mentioned.
    """
    if update.message is None or update.message.text is None:
        return

    message = update.message
    text = message.text

    # In group/supergroup chats, only respond when @mentioned
    if message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        mention = f"@{context.bot.username}"
        if mention not in text:
            return
        user_text = text.replace(mention, "").strip()
    else:
        user_text = text.strip()

    if not user_text:
        await message.reply_text("How can I help you? Ask me anything!")
        return

    thread_id = _thread_id(message)

    # Send typing indicator
    await message.reply_chat_action(ChatAction.TYPING)

    try:
        response = await run_agent(user_text, thread_id)
    except Exception as exc:
        logger.exception("Agent invocation failed for thread %s", thread_id)
        response = f"Sorry, I encountered an error: {exc}"

    # Telegram hard limit is 4096 chars
    if len(response) > 4096:
        response = response[:4093] + "..."

    await message.reply_text(response)


# ---------------------------------------------------------------------------
# DAG command handlers
# ---------------------------------------------------------------------------


async def cmd_branch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /branch -- switch active node to create a branch.

    Usage: /branch <node_id_prefix>
    """
    if update.message is None:
        return

    args = context.args
    if not args:
        await update.message.reply_text("Usage: /branch <node_id_prefix>")
        return

    target_prefix = args[0].lower()
    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    dag_graph = load(jsonl_path)

    # Prefix-match node ID
    matches = [nid for nid in dag_graph.nodes if nid.lower().startswith(target_prefix)]
    if len(matches) == 0:
        await update.message.reply_text(f"No node found matching '{target_prefix}'.")
        return
    if len(matches) > 1:
        match_list = ", ".join(m[:8] for m in matches[:5])
        await update.message.reply_text(f"Ambiguous prefix. Matches: {match_list}")
        return

    target_id = matches[0]
    try:
        switch_active(dag_graph, jsonl_path, target_id)
        write_session(dag_graph, jsonl_path)
        node = dag_graph.nodes[target_id]
        await update.message.reply_text(
            f"Branching from node {target_id[:8]}.\nQ: {node.q[:80]}"
        )
    except ValueError as exc:
        await update.message.reply_text(f"Error: {exc}")


async def cmd_switch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /switch -- alias for /branch."""
    await cmd_branch(update, context)


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /list -- show recent nodes in the conversation.

    Usage: /list [count]
    """
    if update.message is None:
        return

    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    dag_graph = load(jsonl_path)

    if not dag_graph.nodes:
        await update.message.reply_text("No nodes in the conversation graph.")
        return

    # Parse optional count argument
    count = 10
    if context.args:
        try:
            count = int(context.args[0])
        except ValueError:
            pass

    # ULIDs sort chronologically
    sorted_ids = sorted(dag_graph.nodes.keys())[-count:]

    lines: list[str] = []
    for nid in sorted_ids:
        node = dag_graph.nodes[nid]
        active_marker = " *" if nid == dag_graph.active_node else ""
        q_preview = node.q[:60] + ("..." if len(node.q) > 60 else "")
        lines.append(f"{nid[:8]}{active_marker}: {q_preview}")

    header = f"Last {len(sorted_ids)} nodes (* = active):\n"
    response = header + "\n".join(lines)

    if len(response) > 4096:
        response = response[:4093] + "..."

    await update.message.reply_text(response)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the Telegram bot."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment or .env file")

    # Ensure JSONL directory exists
    JSONL_DIR.mkdir(parents=True, exist_ok=True)

    app = ApplicationBuilder().token(token).build()

    # Command handlers (registered before the catch-all message handler)
    app.add_handler(CommandHandler("branch", cmd_branch))
    app.add_handler(CommandHandler("switch", cmd_switch))
    app.add_handler(CommandHandler("list", cmd_list))

    # Message handler (catch-all for non-command text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
