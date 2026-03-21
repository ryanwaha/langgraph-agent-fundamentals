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
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from markdown_it import MarkdownIt
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Message, ReactionTypeEmoji, Update
from telegram.constants import ChatType, ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.context import Context
from agent.dag import Graph, Node, append_node, delete_node, load, maintain, render_topology_svg, switch_active, write_session
from agent.graph import _dag_cache, graph
from agent.state import InputState
from agent.utils import get_message_text

load_dotenv()

logger = logging.getLogger(__name__)

JSONL_DIR = Path("jsonls")

# markdown-it-py parser (CommonMark spec; used by _md_to_html)
_MD = MarkdownIt("commonmark")

# Per-thread asyncio Lock: prevents concurrent agent invocations on the same thread.
# defaultdict(asyncio.Lock) creates a new Lock lazily for each thread_id.
_thread_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# Per-thread pending debounce tasks: maps thread_id → in-flight asyncio.Task.
# When a new message arrives within 800 ms of the previous one, the old task is
# cancelled before the LLM is ever called (OpenClaw-validated window).
_pending_tasks: dict[str, asyncio.Task] = {}

# Pending merge: thread_id → list of parent node IDs.
# Set by /merge, consumed by handle_message on next user text.
_pending_merges: dict[str, list[str]] = {}

# Human-readable labels for tool names (extend when new tools are added)
_TOOL_LABELS: dict[str, str] = {
    "search": "搜尋網頁",
}


def _h(text: str) -> str:
    """HTML-escape text for safe insertion into HTML-mode Telegram messages."""
    return html.escape(text)


def _md_to_html(text: str) -> str:
    """Convert LLM Markdown to Telegram HTML using markdown-it-py (CommonMark).

    markdown-it-py handles nested formatting, links, lists, and fenced code
    blocks correctly.  Output is post-processed to map standard HTML tags to
    Telegram's whitelist: b, i, s, u, code, pre, a, tg-spoiler.
    """
    html_out = _MD.render(text)
    # Map standard HTML → Telegram HTML whitelist
    html_out = re.sub(r"<strong>(.*?)</strong>", r"<b>\1</b>", html_out, flags=re.DOTALL)
    html_out = re.sub(r"<em>(.*?)</em>", r"<i>\1</i>", html_out, flags=re.DOTALL)
    html_out = re.sub(r"<del>(.*?)</del>", r"<s>\1</s>", html_out, flags=re.DOTALL)
    # Headings → bold + newline
    html_out = re.sub(r"<h[1-6]>(.*?)</h[1-6]>", r"<b>\1</b>\n", html_out, flags=re.DOTALL)
    # Paragraphs → text + newline (strip the wrapper)
    html_out = re.sub(r"<p>(.*?)</p>", r"\1\n", html_out, flags=re.DOTALL)
    # List items → bullet
    html_out = re.sub(r"<li>(.*?)</li>", r"• \1\n", html_out, flags=re.DOTALL)
    # Strip list/blockquote wrapper tags
    html_out = re.sub(r"</?(?:ul|ol|blockquote)[^>]*>", "", html_out)
    # Strip any remaining non-whitelisted tags
    html_out = re.sub(
        r"</?(?!(?:b|i|s|u|code|pre|a|tg-spoiler)(?:\s|>|/))[a-zA-Z][^>]*>", "", html_out
    )
    # ~~strikethrough~~ fallback (not in CommonMark preset, handled after render)
    html_out = re.sub(r"~~(.+?)~~", r"<s>\1</s>", html_out, flags=re.DOTALL)
    return html_out.strip()


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
# /view helpers — DAG traversal
# ---------------------------------------------------------------------------


def _walk_up(dag_graph: Graph, start_id: str, limit: int = 3) -> list[str]:
    """Walk parent chain upward from *start_id* (exclusive).

    If start_id itself has multiple parents → return ALL parents, stop.
    Otherwise follow single-parent chain up to *limit* nodes.
    Stop early (inclusive) when hitting a node with multiple parents.
    """
    node = dag_graph.nodes.get(start_id)
    if not node:
        return []
    if len(node.parents) > 1:
        return list(node.parents)

    result: list[str] = []
    cur = node.parents[0] if node.parents else None
    while cur and len(result) < limit:
        result.append(cur)
        n = dag_graph.nodes.get(cur)
        if not n or len(n.parents) != 1:
            break  # root or multi-parent → stop
        cur = n.parents[0]
    return result


def _walk_down(dag_graph: Graph, start_id: str, limit: int = 3) -> list[str]:
    """Walk child chain downward from *start_id* (exclusive).

    If start_id itself has multiple children → return ALL children, stop.
    Otherwise follow single-child chain up to *limit* nodes.
    Stop early (inclusive) when hitting a node with multiple children.
    """
    children = dag_graph.children.get(start_id, [])
    if len(children) > 1:
        return list(children)

    result: list[str] = []
    cur = children[0] if children else None
    while cur and len(result) < limit:
        result.append(cur)
        ch = dag_graph.children.get(cur, [])
        if len(ch) != 1:
            break  # leaf or multi-child → stop
        cur = ch[0]
    return result


def _nearest_branch_below(dag_graph: Graph, node_id: str) -> str | None:
    """BFS downward to find the first node with multiple children."""
    from collections import deque

    queue: deque[str] = deque()
    # seed with children of node_id
    for ch in dag_graph.children.get(node_id, []):
        queue.append(ch)
    visited: set[str] = {node_id}
    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        if len(dag_graph.children.get(nid, [])) > 1:
            return nid
        for ch in dag_graph.children.get(nid, []):
            queue.append(ch)
    return None


def _nearest_merge_above(dag_graph: Graph, node_id: str) -> str | None:
    """Walk single-parent chain upward to find the first node with multiple parents."""
    cur = node_id
    visited: set[str] = set()
    while cur and cur not in visited:
        visited.add(cur)
        n = dag_graph.nodes.get(cur)
        if not n:
            break
        for pid in n.parents:
            p = dag_graph.nodes.get(pid)
            if p and len(p.parents) > 1:
                return pid
            if p and len(p.parents) == 1:
                cur = pid
                break
        else:
            break  # no parents or all checked
    return None


# ---------------------------------------------------------------------------
# /view rendering
# ---------------------------------------------------------------------------

_VIEW_CONTENT_LIMIT = 3000


def _node_markers(dag_graph: Graph, nid: str, active_node: str | None) -> str:
    """Return marker string for a node: ✦ active, ⑂ multi-child, ⑃ multi-parent."""
    parts: list[str] = []
    if nid == active_node:
        parts.append("✦")
    n = dag_graph.nodes.get(nid)
    if n and len(n.parents) > 1:
        parts.append("⑃")
    if len(dag_graph.children.get(nid, [])) > 1:
        parts.append("⑂")
    return " ".join(parts)


def _view_format(
    dag_graph: Graph, node_id: str, thread_id: str, page: int = 0
) -> tuple[str, InlineKeyboardMarkup]:
    """Build the /view message text and inline keyboard for a given node."""
    node = dag_graph.nodes[node_id]
    markers = _node_markers(dag_graph, node_id, dag_graph.active_node)
    marker_str = f" {markers}" if markers else ""

    # --- Current node content (paginated) ---
    content = f"Q: {node.q}\nA: {node.a}"
    total_pages = max(1, (len(content) + _VIEW_CONTENT_LIMIT - 1) // _VIEW_CONTENT_LIMIT)
    page = min(page, total_pages - 1)
    page_content = content[page * _VIEW_CONTENT_LIMIT : (page + 1) * _VIEW_CONTENT_LIMIT]

    lines: list[str] = [f"📍 {node_id[:8]}{marker_str}"]
    lines.append(html.escape(page_content))
    if node.sum:
        lines.append(f"<i>sum: {html.escape(node.sum)}</i>")

    # --- Nearby nodes ---
    up_ids = _walk_up(dag_graph, node_id)
    down_ids = _walk_down(dag_graph, node_id)

    if up_ids or down_ids:
        lines.append("")
        lines.append("──── 鄰近 ────")
        for uid in reversed(up_ids):
            n = dag_graph.nodes.get(uid)
            s = html.escape(n.sum[:60]) if n and n.sum else "—"
            m = _node_markers(dag_graph, uid, dag_graph.active_node)
            m_str = f" {m}" if m else ""
            lines.append(f"↑ {uid[:8]}: {s}{m_str}")
        lines.append(f"► {node_id[:8]}: 當前節點{marker_str}")
        for did in down_ids:
            n = dag_graph.nodes.get(did)
            s = html.escape(n.sum[:60]) if n and n.sum else "—"
            m = _node_markers(dag_graph, did, dag_graph.active_node)
            m_str = f" {m}" if m else ""
            lines.append(f"↓ {did[:8]}: {s}{m_str}")

    # --- Branch / merge points ---
    branch_below = _nearest_branch_below(dag_graph, node_id)
    merge_above = _nearest_merge_above(dag_graph, node_id)
    if branch_below or merge_above:
        lines.append("")
        lines.append("──── 分支點 ────")
        if merge_above:
            pc = len(dag_graph.nodes[merge_above].parents)
            lines.append(f"↑⑃ {merge_above[:8]} ({pc} parents)")
        if branch_below:
            cc = len(dag_graph.children.get(branch_below, []))
            lines.append(f"↓⑂ {branch_below[:8]} ({cc} children)")

    text = "\n".join(lines)

    # --- Keyboard ---
    # Collect all navigable node IDs (deduplicated, preserve order)
    nav_ids: list[str] = []
    seen: set[str] = set()
    for nid in [*reversed(up_ids), *down_ids]:
        if nid not in seen:
            nav_ids.append(nid)
            seen.add(nid)
    for nid in [merge_above, branch_below]:
        if nid and nid not in seen and nid != node_id:
            nav_ids.append(nid)
            seen.add(nid)

    rows: list[list[InlineKeyboardButton]] = []
    # Nav buttons — 3 per row
    for i in range(0, len(nav_ids), 3):
        row = [
            InlineKeyboardButton(
                nid[:8],
                callback_data={"a": "v", "op": "nav", "nid": nid, "tid": thread_id},
            )
            for nid in nav_ids[i : i + 3]
        ]
        rows.append(row)

    # Action row
    action_row = [
        InlineKeyboardButton("🌿 Branch", callback_data={"a": "v", "op": "br", "nid": node_id, "tid": thread_id}),
        InlineKeyboardButton("🗑 Del", callback_data={"a": "v", "op": "del", "nid": node_id, "tid": thread_id}),
        InlineKeyboardButton("✖ Close", callback_data={"a": "v", "op": "x"}),
    ]
    rows.append(action_row)

    # Pagination row
    if total_pages > 1:
        page_row: list[InlineKeyboardButton] = []
        if page > 0:
            page_row.append(InlineKeyboardButton("◀ 上一頁", callback_data={"a": "v", "op": "pg", "nid": node_id, "tid": thread_id, "pg": page - 1}))
        if page < total_pages - 1:
            page_row.append(InlineKeyboardButton("下一頁 ▶", callback_data={"a": "v", "op": "pg", "nid": node_id, "tid": thread_id, "pg": page + 1}))
        rows.append(page_row)

    return text, InlineKeyboardMarkup(rows)


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
    bot: object,
    chat_id: int,
    message_thread_id: int | None,
) -> Message:
    """Stream agent execution via sendMessageDraft (forum topic mode).

    Sends animated draft bubbles while the agent is running, then publishes
    the final answer as a real message.  Returns the published Message so the
    caller can attach an inline keyboard.
    """
    config = RunnableConfig(configurable={"thread_id": thread_id})

    header = f"💬 <b>用戶說：</b>{_h(user_message[:100])}"
    steps: list[str] = []
    current_query: str | None = None
    answer_buf = ""

    def _build(extra: str = "", spoiler_steps: bool = False) -> str:
        parts = [header]
        if steps:
            steps_block = "\n".join(steps)
            parts.append(f"<tg-spoiler>{steps_block}</tg-spoiler>" if spoiler_steps else steps_block)
        if extra:
            parts.append(extra)
        return _truncate("\n\n".join(parts))

    async def _draft(text: str) -> None:
        try:
            await bot.send_message_draft(  # type: ignore[attr-defined]
                chat_id,
                draft_id=1,
                text=text,
                message_thread_id=message_thread_id,
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            logger.exception("send_message_draft failed")

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
            await _draft(_build("🤔 正在思考..."))

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
            await _draft(_build())

        elif evt == "on_tool_end":
            tool_name = event["name"]
            label = _TOOL_LABELS.get(tool_name, tool_name)
            for i in range(len(steps) - 1, -1, -1):
                if steps[i].startswith("⏳"):
                    if tool_name == "search" and current_query:
                        steps[i] = f"✅ 已{label}：「{_h(current_query[:60])}」"
                    else:
                        steps[i] = steps[i].replace("⏳ 正在", "✅ 已完成", 1)
                    break
            current_query = None
            await _draft(_build())

        elif evt == "on_chat_model_stream" and node == "call_model":
            chunk = event["data"].get("chunk")
            if chunk is not None:
                token = get_message_text(chunk)
                if token:
                    answer_buf += token
                    await _draft(_build(_md_to_html(answer_buf)))

    final_answer = answer_buf.strip()
    final_text = (
        _build(_md_to_html(final_answer), spoiler_steps=True)
        if final_answer
        else _build("(無回應)", spoiler_steps=True)
    )
    # Publish the final answer as a real message (replaces the draft bubble)
    final_msg: Message = await bot.send_message(  # type: ignore[attr-defined]
        chat_id,
        text=final_text,
        message_thread_id=message_thread_id,
        parse_mode=ParseMode.HTML,
    )
    return final_msg


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

        # If a /merge is pending, attach parent IDs so graph.py uses them
        pending_merge = _pending_merges.pop(thread_id, None)
        if pending_merge:
            dag = _dag_cache.get(thread_id)
            if dag:
                dag._merge_parents = pending_merge  # type: ignore[attr-defined]

        node_id: str | None = None
        async with _thread_locks[thread_id]:
            try:
                final_msg = await run_agent_streaming(
                    user_text,
                    thread_id,
                    context.bot,
                    message.chat_id,
                    message.message_thread_id,
                )
            except Exception as exc:
                logger.exception("Agent invocation failed for thread %s", thread_id)
                await message.reply_text(f"Sorry, I encountered an error: {exc}")
                return

        # Attach inline keyboard (Regenerate / Branch here)
        kb = _make_reply_keyboard(thread_id, user_text, node_id)
        try:
            await final_msg.edit_reply_markup(reply_markup=kb)
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
        old_msg = query.message
        chat_id_regen: int = old_msg.chat_id
        thread_id_regen: int | None = old_msg.message_thread_id
        # Delete the old answer; new answer comes from run_agent_streaming
        try:
            await old_msg.delete()
        except Exception:
            pass
        async with _thread_locks[thread_id]:
            try:
                final_msg = await run_agent_streaming(
                    user_message,
                    thread_id,
                    context.bot,
                    chat_id_regen,
                    thread_id_regen,
                )
            except Exception as exc:
                await context.bot.send_message(chat_id_regen, text=f"Error: {exc}", message_thread_id=thread_id_regen)
                return
        kb = _make_reply_keyboard(thread_id, user_message, node_id)
        try:
            await final_msg.edit_reply_markup(reply_markup=kb)
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

    elif action == "v":
        op = data.get("op")

        if op == "x":
            # Close the viewer
            try:
                await query.message.delete()
            except Exception:
                pass
            return

        tid: str = data.get("tid", "")
        nid: str = data.get("nid", "")
        if not tid or not nid:
            return
        jsonl_path = _jsonl_path(tid)
        if not jsonl_path.exists():
            await query.answer("No history found.", show_alert=True)
            return
        dag_graph = load(jsonl_path)

        if op == "nav":
            if nid not in dag_graph.nodes:
                await query.answer("Node not found.", show_alert=True)
                return
            text, kb = _view_format(dag_graph, nid, tid)
            try:
                await query.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
            except Exception:
                pass

        elif op == "pg":
            pg = data.get("pg", 0)
            text, kb = _view_format(dag_graph, nid, tid, page=pg)
            try:
                await query.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
            except Exception:
                pass

        elif op == "br":
            # Switch active to this node (branch from here)
            if nid not in dag_graph.nodes:
                await query.answer("Node not found.", show_alert=True)
                return
            try:
                switch_active(dag_graph, jsonl_path, nid)
                write_session(dag_graph, jsonl_path)
                if tid in _dag_cache:
                    _dag_cache[tid].active_node = nid
                await query.answer(f"✅ Active → {nid[:8]}", show_alert=False)
            except Exception as exc:
                await query.answer(f"Error: {exc}", show_alert=True)
                return
            # Re-render view at this node
            dag_graph = load(jsonl_path)
            text, kb = _view_format(dag_graph, nid, tid)
            try:
                await query.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
            except Exception:
                pass

        elif op == "del":
            if nid not in dag_graph.nodes:
                await query.answer("Node not found.", show_alert=True)
                return
            node = dag_graph.nodes[nid]
            try:
                delete_node(dag_graph, jsonl_path, nid)
            except Exception as exc:
                await query.answer(f"Error: {exc}", show_alert=True)
                return
            # If deleted node was active, retreat to parent
            if dag_graph.active_node == nid:
                dag_graph.active_node = node.parents[0] if node.parents else None
            write_session(dag_graph, jsonl_path)
            if tid in _dag_cache:
                _dag_cache[tid] = dag_graph
            await query.answer(f"🗑 Deleted {nid[:8]}", show_alert=False)
            # Re-render at parent (or just close if no parent)
            retreat_id = node.parents[0] if node.parents else None
            if retreat_id and retreat_id in dag_graph.nodes:
                dag_graph = load(jsonl_path)
                text, kb = _view_format(dag_graph, retreat_id, tid)
                try:
                    await query.message.edit_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)
                except Exception:
                    pass
            else:
                try:
                    await query.message.delete()
                except Exception:
                    pass


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


async def cmd_merge(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /merge -- begin a merge of two or more branches.

    Usage: /merge <id1>,<id2>[,<id3>...]   or   /merge <id1> <id2> ...
    After this, send the merge query as the next message.
    """
    if update.message is None:
        return

    args = context.args
    if not args:
        await update.message.reply_text("Usage: /merge <id1>,<id2> 或 /merge <id1> <id2>")
        return

    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    dag_graph = load(jsonl_path)

    # Parse parent IDs: support comma-separated or space-separated
    raw_ids: list[str] = []
    for arg in args:
        raw_ids.extend(arg.split(","))
    raw_ids = [r.strip() for r in raw_ids if r.strip()]

    if len(raw_ids) < 2:
        await update.message.reply_text("至少需要兩個父節點。")
        return

    # Resolve each prefix
    resolved: list[str] = []
    for prefix in raw_ids:
        nid, err = _match_node(dag_graph, prefix)
        if err:
            await update.message.reply_text(err)
            return
        assert nid is not None
        node = dag_graph.nodes[nid]
        if node.compressed:
            hint = node.parents[0][:8] if node.parents else "?"
            await update.message.reply_text(f"{nid[:8]} 是壓縮節點，請改用其父節點: {hint}")
            return
        resolved.append(nid)

    _pending_merges[thread_id] = resolved
    labels = ", ".join(r[:8] for r in resolved)
    await update.message.reply_text(f"已選定父節點 {labels} — 請發送合併問句")


async def cmd_view(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /view -- interactive DAG node viewer.

    Usage: /view [node_id_prefix]   (defaults to active node)
    """
    if update.message is None:
        return

    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    dag_graph = load(jsonl_path)

    # Determine target node
    args = context.args
    if args:
        target_id, err = _match_node(dag_graph, args[0])
        if err:
            await update.message.reply_text(err)
            return
    else:
        target_id = dag_graph.active_node
        if not target_id:
            await update.message.reply_text("No active node.")
            return

    text, kb = _view_format(dag_graph, target_id, thread_id)
    await update.message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML)


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
        BotCommand("view",     "互動式節點瀏覽器"),
        BotCommand("merge",    "合併分支（選定父節點後發送問句）"),
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
    app.add_handler(CommandHandler("view",     cmd_view))
    app.add_handler(CommandHandler("merge",    cmd_merge))
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
