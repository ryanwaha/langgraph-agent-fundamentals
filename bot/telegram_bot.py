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
from agent.dag import delete_node, load, maintain, render_topology_svg, switch_active, write_session
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


def _resolve_node_id(dag_graph, prefix: str) -> str | None:
    """Resolve a ULID prefix to a full node ID. Returns None and sets reply on ambiguity/miss."""
    matches = [nid for nid in dag_graph.nodes if nid.lower().startswith(prefix.lower())]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        return None
    return None  # ambiguous — caller checks len(matches) separately


def _match_node(dag_graph, prefix: str) -> tuple[str | None, str | None]:
    """Return (node_id, error_message). Exactly one match → (id, None); else (None, msg)."""
    matches = [nid for nid in dag_graph.nodes if nid.lower().startswith(prefix.lower())]
    if len(matches) == 0:
        return None, f"No node found matching '{prefix}'."
    if len(matches) > 1:
        sample = ", ".join(m[:8] for m in matches[:5])
        return None, f"Ambiguous prefix. Matches: {sample}"
    return matches[0], None


# ---------------------------------------------------------------------------
# Graph invocation
# ---------------------------------------------------------------------------


async def run_agent(user_message: str, thread_id: str) -> str:
    """Invoke the LangGraph agent and return its answer.

    DAG lifecycle (load → build context → append node → write session)
    is handled inside the agent graph nodes.
    """
    config = RunnableConfig(configurable={"thread_id": thread_id})
    result = await graph.ainvoke(
        InputState(messages=[HumanMessage(content=user_message)]),
        config=config,
        context=Context(),
    )
    return get_message_text(result["messages"][-1])


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
    text: str = message.text  # type: ignore[assignment]  # guarded by None check above

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

    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    dag_graph = load(jsonl_path)

    target_id, err = _match_node(dag_graph, args[0])
    if err:
        await update.message.reply_text(err)
        return
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


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status -- show thread info and active node."""
    if update.message is None:
        return

    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    dag_graph = load(jsonl_path)
    active = dag_graph.active_node[:8] if dag_graph.active_node else "None"
    await update.message.reply_text(
        f"thread: {thread_id}\nnodes: {len(dag_graph.nodes)}\nactive: {active}"
    )


async def cmd_show(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /show -- show the active node's path to root."""
    if update.message is None:
        return

    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    dag_graph = load(jsonl_path)

    if dag_graph.active_node is None:
        await update.message.reply_text("No active node.")
        return

    nodes = dag_graph.path_to_root(dag_graph.active_node)
    if not nodes:
        await update.message.reply_text("(active path is empty)")
        return

    lines: list[str] = []
    for node in nodes:
        lines.append(f"[{node.id[:8]}] Q: {node.q}")
        if node.a:
            lines.append(f"         A: {node.a[:120]}{'...' if len(node.a) > 120 else ''}")

    response = "\n".join(lines)
    if len(response) > 4096:
        response = response[:4093] + "..."

    await update.message.reply_text(response)


async def cmd_paths(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /paths -- show all paths from active node to roots."""
    if update.message is None:
        return

    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    dag_graph = load(jsonl_path)
    paths = dag_graph.active_paths()

    if not paths:
        await update.message.reply_text("No active paths.")
        return

    lines: list[str] = []
    for idx, path_nodes in enumerate(paths, start=1):
        path_str = " -> ".join(n.id[:8] for n in path_nodes)
        lines.append(f"path {idx}: {path_str}")

    response = "\n".join(lines)
    if len(response) > 4096:
        response = response[:4093] + "..."

    await update.message.reply_text(response)


async def cmd_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /delete -- soft-delete a node by ULID prefix.

    Usage: /delete <node_id_prefix>
    """
    if update.message is None:
        return

    args = context.args
    if not args:
        await update.message.reply_text("Usage: /delete <node_id_prefix>")
        return

    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    dag_graph = load(jsonl_path)

    target_id, err = _match_node(dag_graph, args[0])
    if err:
        await update.message.reply_text(err)
        return

    assert target_id is not None
    node = dag_graph.nodes[target_id]
    child_count = len(dag_graph.children.get(target_id, []))
    warnings: list[str] = []
    if len(node.parents) > 1:
        warnings.append(f"[warn] {target_id[:8]} is a merge node ({len(node.parents)} parents).")
    if child_count > 1:
        warnings.append(f"[warn] {target_id[:8]} is a branch node ({child_count} children).")
    elif child_count == 1:
        warnings.append(f"[warn] {target_id[:8]} has a child node.")

    try:
        result = delete_node(dag_graph, jsonl_path, target_id)
    except Exception as exc:
        await update.message.reply_text(f"Error: {exc}")
        return

    if dag_graph.active_node == target_id:
        dag_graph.active_node = node.parents[0] if node.parents else None
    write_session(dag_graph, jsonl_path)

    if result["already_deleted"]:
        msg = f"{target_id[:8]} is already deleted."
    else:
        msg = f"Deleted: {target_id[:8]}"
    if warnings:
        msg = "\n".join(warnings) + "\n" + msg

    await update.message.reply_text(msg)


async def cmd_render(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /render -- render DAG topology to SVG and send as file."""
    if update.message is None:
        return

    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    dag_graph = load(jsonl_path)

    try:
        output_path = render_topology_svg(dag_graph, jsonl_path)
    except Exception as exc:
        await update.message.reply_text(f"Render failed: {exc}")
        return

    with open(output_path, "rb") as f:
        await update.message.reply_document(document=f, filename=output_path.name)


async def cmd_maintain(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /maintain -- compact, reorder, and rebuild the JSONL index."""
    if update.message is None:
        return

    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    try:
        maintain(jsonl_path)
    except Exception as exc:
        await update.message.reply_text(f"Maintain failed: {exc}")
        return

    await update.message.reply_text("Maintained: compact → reorder → rebuild_index")


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
    app.add_handler(CommandHandler("branch",   cmd_branch))
    app.add_handler(CommandHandler("switch",   cmd_switch))
    app.add_handler(CommandHandler("list",     cmd_list))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("show",     cmd_show))
    app.add_handler(CommandHandler("paths",    cmd_paths))
    app.add_handler(CommandHandler("delete",   cmd_delete))
    app.add_handler(CommandHandler("render",   cmd_render))
    app.add_handler(CommandHandler("maintain", cmd_maintain))

    # Message handler (catch-all for non-command text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
