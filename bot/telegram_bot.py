"""Telegram bot entry point.

Bridges Telegram messages to the LangGraph ReAct agent with DAG-based
conversation persistence.  Each chat (or forum topic) gets its own JSONL
file, enabling branching conversation history.
"""

import asyncio
import html
import logging
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Message, ReactionTypeEmoji, Update
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.context import Context
from agent.dag import delete_node, load, maintain, render_topology_svg, switch_active, write_session
from agent.graph import _dag_cache, graph
from agent.state import InputState
from agent.utils import get_message_text

load_dotenv()

logger = logging.getLogger(__name__)

JSONL_DIR = Path("jsonls")

# Minimum seconds between Telegram editMessage calls (flood limit ~1/sec per chat)
_EDIT_INTERVAL = 1.2

# Per-thread asyncio Lock: prevents concurrent agent invocations on the same thread.
# defaultdict(asyncio.Lock) creates a new Lock lazily for each thread_id.
_thread_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# Per-thread pending debounce tasks: maps thread_id → in-flight asyncio.Task.
# When a new message arrives within 800 ms of the previous one, the old task is
# cancelled before the LLM is ever called (OpenClaw-validated window).
_pending_tasks: dict[str, asyncio.Task] = {}

# Human-readable labels for tool names (extend when new tools are added)
_TOOL_LABELS: dict[str, str] = {
    "search": "搜尋網頁",
}


def _h(text: str) -> str:
    """HTML-escape text for safe insertion into HTML-mode Telegram messages."""
    return html.escape(text)


def _md_to_html(text: str) -> str:
    """Convert common LLM Markdown to Telegram HTML (parse_mode=HTML).

    Handles: fenced code blocks, inline code, bold, italic, headers,
    strikethrough.  Non-code text is HTML-escaped first so that angle
    brackets / ampersands in prose are always safe.
    """
    result: list[str] = []
    # Split on fenced code blocks (```...```) — captured so odd-indexed parts are code.
    parts = re.split(r"(```[^\n]*\n[\s\S]*?```)", text)
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Fenced code block — extract optional language tag and content.
            m = re.match(r"```[^\n]*\n([\s\S]*?)```", part)
            code = html.escape(m.group(1).rstrip()) if m else html.escape(part)
            result.append(f"<pre><code>{code}</code></pre>")
        else:
            # Prose — escape HTML first, then apply inline markdown patterns.
            s = html.escape(part)
            # Inline code (content already escaped).
            s = re.sub(r"`([^`\n]+)`", lambda m: f"<code>{m.group(1)}</code>", s)
            # Bold (**...**)
            s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s, flags=re.DOTALL)
            # Italic (*...*)  — only single asterisk remaining after bold pass.
            s = re.sub(r"\*([^*\n]+?)\*", r"<i>\1</i>", s)
            # Strikethrough
            s = re.sub(r"~~(.+?)~~", r"<s>\1</s>", s, flags=re.DOTALL)
            # ATX headers (# / ## / ###…) → bold line
            s = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", s, flags=re.MULTILINE)
            result.append(s)
    return "".join(result)


def _truncate(text: str, limit: int = 4096) -> str:
    return text[: limit - 3] + "..." if len(text) > limit else text


def _make_reply_keyboard(
    thread_id: str, user_message: str, node_id: str | None
) -> InlineKeyboardMarkup:
    """Build the inline keyboard attached to every AI reply."""
    buttons = [
        InlineKeyboardButton(
            "🔄 Regenerate",
            callback_data={"a": "regen", "tid": thread_id, "q": user_message},
        )
    ]
    if node_id:
        buttons.append(
            InlineKeyboardButton(
                "🌿 Branch here",
                callback_data={"a": "branch", "tid": thread_id, "nid": node_id},
            )
        )
    return InlineKeyboardMarkup([buttons])


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


async def run_agent_streaming(
    user_message: str,
    thread_id: str,
    status_msg: Message,
) -> str:
    """Stream agent execution, updating status_msg with live CoT progress.

    Displays step-by-step progress (thinking → searching → answer) by
    editing the status message as astream_events fires.  Returns the
    final answer string.
    """
    config = RunnableConfig(configurable={"thread_id": thread_id})

    header = f"💬 <b>用戶說：</b>{_h(user_message[:100])}"
    steps: list[str] = []          # one entry per tool call (⏳ → ✅)
    current_query: str | None = None
    answer_buf = ""
    last_edit = 0.0

    def _build(extra: str = "", spoiler_steps: bool = False) -> str:
        parts = [header]
        if steps:
            steps_block = "\n".join(steps)
            parts.append(f"<tg-spoiler>{steps_block}</tg-spoiler>" if spoiler_steps else steps_block)
        if extra:
            parts.append(extra)
        return _truncate("\n\n".join(parts))

    async def _edit(text: str, force: bool = False) -> None:
        nonlocal last_edit
        now = time.monotonic()
        if not force and now - last_edit < _EDIT_INTERVAL:
            return
        try:
            await status_msg.edit_text(text, parse_mode=ParseMode.HTML)
            last_edit = now
        except BadRequest:
            pass  # "message is not modified" — safe to ignore
        except Exception:
            pass

    async for event in graph.astream_events(
        InputState(messages=[HumanMessage(content=user_message)]),
        config=config,
        context=Context(),
        version="v2",
    ):
        evt: str = event["event"]
        node: str = event.get("metadata", {}).get("langgraph_node", "")

        if evt == "on_chat_model_start" and node == "call_model":
            answer_buf = ""
            await _edit(_build("🤔 正在思考..."))

        elif evt == "on_tool_start":
            tool_name: str = event["name"]
            tool_input: dict = event["data"].get("input") or {}
            label = _TOOL_LABELS.get(tool_name, tool_name)
            if tool_name == "search":
                current_query = str(tool_input.get("query", ""))
                steps.append(f"⏳ 正在{label}：「{_h(current_query[:60])}」")
            else:
                current_query = None
                steps.append(f"⏳ 正在執行：{_h(label)}")
            await _edit(_build())

        elif evt == "on_tool_end":
            tool_name = event["name"]
            label = _TOOL_LABELS.get(tool_name, tool_name)
            # Replace the last ⏳ line for this tool with ✅
            for i in range(len(steps) - 1, -1, -1):
                if steps[i].startswith("⏳"):
                    if tool_name == "search" and current_query:
                        steps[i] = f"✅ 已{label}：「{_h(current_query[:60])}」"
                    else:
                        steps[i] = steps[i].replace("⏳ 正在", "✅ 已完成", 1)
                    break
            current_query = None
            await _edit(_build())

        elif evt == "on_chat_model_stream" and node == "call_model":
            chunk = event["data"].get("chunk")
            if chunk is not None:
                token = get_message_text(chunk)
                if token:
                    answer_buf += token
                    await _edit(_build(_md_to_html(answer_buf)))

        # Ignore all events from the "summarize" node (DAG summarization LLM call)

    final_answer = answer_buf.strip()
    # Final unconditional edit: answer in MD→HTML, CoT steps collapsed into spoiler.
    final_text = (
        _build(_md_to_html(final_answer), spoiler_steps=True)
        if final_answer
        else _build("(無回應)", spoiler_steps=True)
    )
    await _edit(final_text, force=True)
    return final_answer


# ---------------------------------------------------------------------------
# Telegram event handlers
# ---------------------------------------------------------------------------


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming Telegram messages.

    In private chats, always responds.
    In group chats, only responds when @mentioned.

    Implements two concurrency safeguards:
    - Debouncing (800 ms): rapid successive messages cancel the previous pending
      invocation so only the last message in a burst reaches the LLM.
    - Per-thread Lock: even if two tasks somehow overlap (e.g. a message and a
      Regenerate button press), the second waits for the first to finish.
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

    # Cancel any in-flight debounce task for this thread
    existing = _pending_tasks.get(thread_id)
    if existing and not existing.done():
        existing.cancel()

    async def _run() -> None:
        # Debounce window: if another message arrives within 800 ms, this task
        # will be cancelled before the sleep completes.
        await asyncio.sleep(0.8)

        # React 👀 to acknowledge receipt
        try:
            await message.set_reaction([ReactionTypeEmoji("👀")])
        except Exception:
            pass

        # Send initial placeholder message that will be progressively edited
        init_text = f"💬 <b>用戶說：</b>{_h(user_text[:100])}\n\n🤔 正在思考..."
        status_msg = await message.reply_text(init_text, parse_mode=ParseMode.HTML)

        async with _thread_locks[thread_id]:
            try:
                await run_agent_streaming(user_text, thread_id, status_msg)
            except Exception as exc:
                logger.exception("Agent invocation failed for thread %s", thread_id)
                await status_msg.edit_text(f"Sorry, I encountered an error: {exc}")
                return

        # Attach inline keyboard (Regenerate / Branch here)
        dag = _dag_cache.get(thread_id)
        node_id = dag.active_node if dag else None
        kb = _make_reply_keyboard(thread_id, user_text, node_id)
        try:
            await status_msg.edit_reply_markup(reply_markup=kb)
        except Exception:
            pass

        # React ✅ on success
        try:
            await message.set_reaction([ReactionTypeEmoji("✅")])
        except Exception:
            pass

    task = asyncio.create_task(_run())
    _pending_tasks[thread_id] = task


# ---------------------------------------------------------------------------
# Inline keyboard callback handler
# ---------------------------------------------------------------------------


async def cmd_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses (Regenerate / Branch here)."""
    query = update.callback_query
    if query is None:
        return
    await query.answer()  # dismiss Telegram spinner

    data: dict = query.data  # arbitrary_callback_data delivers the original dict
    action = data.get("a")
    thread_id: str = data.get("tid", "")

    if action == "regen":
        user_message: str = data.get("q", "")
        if not user_message or not thread_id:
            return
        await query.edit_message_text(
            f"💬 <b>用戶說：</b>{_h(user_message[:100])}\n\n🔄 正在重新思考...",
            parse_mode=ParseMode.HTML,
        )
        status_msg = query.message
        async with _thread_locks[thread_id]:
            try:
                await run_agent_streaming(user_message, thread_id, status_msg)
            except Exception as exc:
                await status_msg.edit_text(f"Error: {exc}")
                return
        dag = _dag_cache.get(thread_id)
        node_id = dag.active_node if dag else None
        kb = _make_reply_keyboard(thread_id, user_message, node_id)
        try:
            await status_msg.edit_reply_markup(reply_markup=kb)
        except Exception:
            pass

    elif action == "branch":
        node_id: str = data.get("nid", "")
        if not node_id or not thread_id:
            return
        jsonl_path = _jsonl_path(thread_id)
        if not jsonl_path.exists():
            await query.answer("No history found.", show_alert=True)
            return
        dag_graph = load(jsonl_path)
        target_id, err = _match_node(dag_graph, node_id)
        if err:
            await query.answer(err, show_alert=True)
            return
        try:
            switch_active(dag_graph, jsonl_path, target_id)
            write_session(dag_graph, jsonl_path)
            if thread_id in _dag_cache:
                _dag_cache[thread_id].active_node = target_id
            await query.answer(f"✅ Branched to {target_id[:8]}", show_alert=False)
        except Exception as exc:
            await query.answer(f"Error: {exc}", show_alert=True)


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


async def _post_init(app) -> None:  # type: ignore[type-arg]
    """Register bot commands so Telegram shows the / menu."""
    await app.bot.set_my_commands([
        BotCommand("branch",   "切換到指定節點（開始新分支）"),
        BotCommand("switch",   "/branch 別名"),
        BotCommand("list",     "列出最近的對話節點"),
        BotCommand("status",   "顯示目前會話狀態"),
        BotCommand("show",     "顯示當前節點到根的路徑"),
        BotCommand("paths",    "顯示所有分支路徑"),
        BotCommand("delete",   "軟刪除指定節點"),
        BotCommand("render",   "渲染 DAG 拓撲 SVG"),
        BotCommand("maintain", "整理 JSONL 索引"),
    ])


def main() -> None:
    """Start the Telegram bot."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment or .env file")

    # Ensure JSONL directory exists
    JSONL_DIR.mkdir(parents=True, exist_ok=True)

    app = (
        ApplicationBuilder()
        .token(token)
        .arbitrary_callback_data(True)
        .post_init(_post_init)
        .build()
    )

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

    # Inline keyboard callbacks (Regenerate / Branch here)
    app.add_handler(CallbackQueryHandler(cmd_callback))

    # Message handler (catch-all for non-command text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
