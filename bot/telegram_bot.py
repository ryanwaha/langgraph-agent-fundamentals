"""Telegram bot entry point.

Bridges Telegram messages to the LangGraph ReAct agent with DAG-based
conversation persistence.  Each chat (or forum topic) gets its own JSONL
file, enabling branching conversation history.
"""

import asyncio
import html
import io
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

from langgraph.types import Command

from agent.context import Context
from agent.logging_config import setup_logging
from agent.dag import Graph, Node, append_node, delete_node, generate_id, load, maintain, render_topology_png, switch_active, write_session
from agent.graph import flush_all_sessions, flush_session, graph, start_session_flusher
from agent.state import InputState
from agent.utils import get_message_text

load_dotenv()

logger = logging.getLogger(__name__)

JSONL_DIR = Path("jsonls")

# markdown-it-py parser (CommonMark + tables plugin; used by _md_to_html)
_MD = MarkdownIt("commonmark").enable("table")

# Per-thread asyncio Lock: prevents concurrent agent invocations on the same thread.
# defaultdict(asyncio.Lock) creates a new Lock lazily for each thread_id.
_thread_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# Per-thread pending debounce tasks: maps thread_id → in-flight asyncio.Task.
# When a new message arrives within 800 ms of the previous one, the old task is
# cancelled before the LLM is ever called (OpenClaw-validated window).
_pending_tasks: dict[str, asyncio.Task] = {}

# Pending merge: dag_thread_id → list of parent node IDs.
# Set by /merge, consumed by handle_message on next user text.
_pending_merges: dict[str, list[str]] = {}

# Pending interrupt: dag_thread_id → invoke_id.
# Set when the agent calls ask_user (interrupt); consumed on the next user reply.
_pending_invoke: dict[str, str] = {}

# Human-readable labels for tool names (extend when new tools are added)
_TOOL_LABELS: dict[str, str] = {
    "search": "搜尋網頁",
    "ask_user": "詢問使用者",
}


def _h(text: str) -> str:
    """HTML-escape text for safe insertion into HTML-mode Telegram messages."""
    return html.escape(text)


def _classify_error(exc: BaseException) -> str:
    """Return a user-friendly error description based on exception type/message."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timeout" in msg or "timed out" in msg:
        return "網路逾時，請稍後再試"
    if "connect" in name or "connection" in msg or "network" in msg:
        return "網路連線錯誤，請稍後再試"
    if "429" in msg or "quota" in msg or "resource exhausted" in msg or "rate limit" in msg:
        return "API 配額超限，請稍後再試"
    if "503" in msg or "unavailable" in msg or "overloaded" in msg:
        return "模型服務暫時無法使用，請稍後再試"
    if "401" in msg or "403" in msg or "unauthorized" in msg or "forbidden" in msg:
        return "API 驗證錯誤，請檢查設定"
    return f"發生錯誤：{type(exc).__name__}"


def _dw(s: str) -> int:
    """Display width of a string: CJK full-width chars count as 2, others as 1."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _ljust_dw(s: str, width: int) -> str:
    """Left-justify s to display width, padding with spaces."""
    return s + " " * max(0, width - _dw(s))


def _table_to_pre(html_table: str) -> str:
    """Convert an HTML <table> block to a monospace <pre> block for Telegram.

    Telegram's HTML mode does not support <table>, so we render the cells as
    space-aligned plain text inside a <pre> tag.

    Example input (from markdown-it-py):
        <table><thead><tr><th>Name</th><th>Score</th></tr></thead>
               <tbody><tr><td>Alice</td><td>95</td></tr></tbody></table>

    Example output:
        <pre>Name   Score
        -----  -----
        Alice  95</pre>
    """
    # Extract all rows: each <tr> becomes a list of cell texts
    rows: list[list[str]] = []
    is_header_row: list[bool] = []
    for tr_match in re.finditer(r"<tr>(.*?)</tr>", html_table, re.DOTALL):
        row_html = tr_match.group(1)
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.DOTALL)
        # Strip any nested tags from cell text
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if cells:
            rows.append(cells)
            is_header_row.append("<th" in row_html)

    if not rows:
        return ""

    # Determine column widths
    col_count = max(len(r) for r in rows)
    col_widths = [0] * col_count
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], _dw(cell))

    lines: list[str] = []
    for idx, (row, is_header) in enumerate(zip(rows, is_header_row)):
        padded = [_ljust_dw(cell, col_widths[i]) for i, cell in enumerate(row)]
        lines.append("  ".join(padded).rstrip())
        # Insert separator after the header row
        if is_header:
            sep = ["-" * col_widths[i] for i in range(len(row))]
            lines.append("  ".join(sep).rstrip())

    return "<pre>" + "\n".join(lines) + "</pre>"


def _extract_block_math(text: str) -> tuple[str, list[str]]:
    """Strip $$...$$ display math from text and return (cleaned_text, [formulas])."""
    formulas: list[str] = []

    def _collect(m: re.Match) -> str:
        formulas.append(m.group(1).strip())
        return ""

    cleaned = re.sub(r"\$\$(.*?)\$\$", _collect, text, flags=re.DOTALL)
    return cleaned.strip(), formulas


def _render_formula_png(formula: str) -> io.BytesIO:
    """Render a LaTeX display-math formula to a PNG image using matplotlib.mathtext."""
    import matplotlib  # lazy import — only needed when formulas are present
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 1.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, f"${formula}$", size=18, ha="center", va="center",
            transform=ax.transAxes)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, pad_inches=0.3)
    plt.close(fig)
    buf.seek(0)
    return buf


def _md_to_html(text: str) -> str:
    """Convert LLM Markdown to Telegram HTML using markdown-it-py (CommonMark).

    markdown-it-py handles nested formatting, links, lists, and fenced code
    blocks correctly.  Output is post-processed to map standard HTML tags to
    Telegram's whitelist: b, i, s, u, code, pre, a, tg-spoiler.
    """
    # --- LaTeX handling ---
    # $$...$$ must be stashed BEFORE markdown rendering and BEFORE inline $...$
    # conversion, because:
    #   1. markdown-it-py would escape any injected HTML tags.
    #   2. The inline regex \$([^$]+)\$ can accidentally match the inner part of
    #      $$formula$$.
    # Strategy: replace each $$...$$ with a unique control-char token (\x02Fn\x02)
    # that markdown-it-py passes through verbatim; restore as <code> after rendering.
    # Final calls arrive with no $$...$$ (already extracted for image sending), so
    # this loop is a no-op for them.
    _math_stash: list[str] = []

    def _stash(m: re.Match) -> str:
        _math_stash.append(f"<code>{_h(m.group(1).strip())}</code>")
        return f"\x02F{len(_math_stash) - 1}\x02"

    text = re.sub(r"\$\$(.*?)\$\$", _stash, text, flags=re.DOTALL)

    # Convert remaining inline $...$ to Unicode via unicodeit.
    import unicodeit  # lazy import

    # Aliases and wrappers that unicodeit doesn't know.
    _LATEX_PRE = [
        (re.compile(r"\\dots\b"),                    r"\\ldots"),       # \dots → \ldots (unicodeit knows \ldots → …)
        (re.compile(r"\\math(?:rm|bf|it|cal)\{([^}]*)\}"), r"\1"),      # \mathrm{x} → x
        (re.compile(r"\\text\{([^}]*)\}"),           r"\1"),            # \text{foo} → foo
        (re.compile(r"\\operatorname\{([^}]*)\}"),   r"\1"),            # \operatorname{sin} → sin
    ]

    def _pre_process(expr: str) -> str:
        for pat, repl in _LATEX_PRE:
            expr = pat.sub(repl, expr)
        return expr

    def _stash_inline(m: re.Match) -> str:
        raw = m.group(1)
        converted = unicodeit.replace(_pre_process(raw))
        # Strip leftover LaTeX grouping braces (e.g. √{ρ} → √ρ)
        converted = converted.replace("{", "").replace("}", "")
        # Fall back to <code> only if unknown commands (\...) remain
        if "\\" in converted:
            repl = f"<code>{_h(raw)}</code>"
        else:
            repl = f"<b><i>{_h(converted)}</i></b>"
        _math_stash.append(repl)
        return f"\x02F{len(_math_stash) - 1}\x02"

    text = re.sub(r"\$([^$\n]+)\$", _stash_inline, text)

    # Fix CommonMark flanking delimiter failures caused by CJK full-width punctuation.
    #
    # Two failure modes, both fixed by inserting ZWS (U+200B, category Cf — neither
    # Unicode whitespace Zs nor Unicode punctuation P*) adjacent to the delimiter:
    #
    # 1. CLOSING delimiter preceded by CJK punct, e.g. **text（content）**的說明
    #    Right-flanking rule: if preceded by punct, must be followed by whitespace or
    #    punct.  「的」 is neither → fails.  ZWS inserted BEFORE ** makes the delimiter
    #    "preceded by non-punct" → right-flanking succeeds unconditionally.
    #    Pattern: CJK_PUNCT → * (any CJK closing/neutral punct before a delimiter)
    #
    # 2. OPENING delimiter followed by CJK open bracket, e.g. 屬於**「統計...」**
    #    Left-flanking rule: if followed by punct, must be preceded by whitespace or
    #    punct.  「於」 is neither → fails.  ZWS inserted AFTER ** makes the delimiter
    #    "followed by non-punct" → left-flanking succeeds unconditionally.
    #    Pattern: * → CJK_OPEN_PUNCT (only opening brackets, NOT 。！？etc. to avoid
    #    breaking cases like **99.98%**。 where ZWS after closing ** invalidates it)
    _CJK_PUNCT      = r"[（）「」【】〔〕『』，。！？：；、﹐﹑﹒﹔﹕]"
    _CJK_OPEN_PUNCT = r"[（「【〔『]"
    _ZWS = "\u200b"
    text = re.sub(rf"({_CJK_PUNCT})([*_~`])",      rf"\1{_ZWS}\2", text)  # rule 1
    text = re.sub(rf"([*_~`])({_CJK_OPEN_PUNCT})", rf"\1{_ZWS}\2", text)  # rule 2

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
    # Tables → monospace <pre> blocks (Telegram doesn't support <table>)
    html_out = re.sub(r"<table>.*?</table>", lambda m: _table_to_pre(m.group(0)), html_out, flags=re.DOTALL)
    # Strip any remaining non-whitelisted tags
    html_out = re.sub(
        r"</?(?!(?:b|i|s|u|code|pre|a|tg-spoiler)(?:\s|>|/))[a-zA-Z][^>]*>", "", html_out
    )
    # ~~strikethrough~~ fallback (not in CommonMark preset, handled after render)
    html_out = re.sub(r"~~(.+?)~~", r"<s>\1</s>", html_out, flags=re.DOTALL)
    # Restore stashed $$...$$ tokens as <code> blocks.
    for i, repl in enumerate(_math_stash):
        html_out = html_out.replace(f"\x02F{i}\x02", repl)
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


async def run_agent_streaming(
    user_message: str,
    dag_thread_id: str,
    invoke_id: str,
    bot: object,
    chat_id: int,
    message_thread_id: int | None,
    *,
    resume_value: str | None = None,
    merge_parents: list[str] | None = None,
) -> tuple[Message | None, str | None]:
    """Stream agent execution via sendMessageDraft.

    On normal completion returns (final_message, last_node_id).
    If the agent interrupts (e.g. ask_user tool), sends the question to the
    user and returns (None, None) — the invoke_id is stored in _pending_invoke
    so the next user reply resumes it.

    Args:
        user_message:   Original user text (used for the draft header).
        dag_thread_id:  Telegram thread ID — keys the JSONL/DAG files.
        invoke_id:      Per-invoke ULID — keys the LangGraph checkpointer.
        resume_value:   If set, resumes a paused interrupt instead of new invoke.
        merge_parents:  DAG parent IDs for /merge, forwarded via configurable.
    """
    configurable: dict = {"thread_id": invoke_id, "dag_thread_id": dag_thread_id}
    if merge_parents:
        configurable["merge_parents"] = merge_parents
    config = RunnableConfig(configurable=configurable)

    input_data = (
        Command(resume=resume_value)
        if resume_value is not None
        else InputState(messages=[HumanMessage(content=user_message)])
    )

    header = f"💬 <b>用戶說：</b>{_h(user_message[:100])}"
    steps: list[str] = []
    current_query: str | None = None
    answer_buf = ""
    thinking_buf = ""
    thinking_step_idx: int = -1
    last_node_id: str | None = None

    def _build(extra: str = "") -> str:
        parts = [header]
        if steps:
            steps_block = "\n".join(steps)
            parts.append(f"<blockquote expandable>{steps_block}</blockquote>")
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
        input_data,
        config=config,
        context=Context(),
        version="v2",
    ):
        evt: str = event["event"]
        node: str = event.get("metadata", {}).get("langgraph_node", "")

        if evt == "on_chat_model_start" and node == "call_model":
            answer_buf = ""
            thinking_buf = ""
            thinking_step_idx = -1
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
                content = chunk.content
                blocks = content if isinstance(content, list) else ([{"type": "text", "text": content}] if content else [])
                for block in blocks:
                    btype = block.get("type", "text") if isinstance(block, dict) else "text"
                    btext = (block.get("thinking", "") if btype == "thinking" else block.get("text", "")) if isinstance(block, dict) else str(block)
                    if not btext:
                        continue
                    if btype == "thinking":
                        if not thinking_buf:
                            steps.append("🧠 正在推理...")
                            thinking_step_idx = len(steps) - 1
                        thinking_buf += btext
                        steps[thinking_step_idx] = f"🧠 正在推理（{len(thinking_buf)} 字）..."
                        await _draft(_build())
                    else:
                        if thinking_step_idx >= 0:
                            preview = _h(thinking_buf[:300])
                            suffix = "..." if len(thinking_buf) > 300 else ""
                            steps[thinking_step_idx] = f"🧠 {preview}{suffix}"
                            thinking_step_idx = -1
                        answer_buf += btext
                        await _draft(_build(_md_to_html(answer_buf)))

        elif evt == "on_chain_end" and node == "":
            # Top-level graph completion: capture last_node_id from output state
            output = event["data"].get("output") or {}
            last_node_id = output.get("last_node_id") or last_node_id

    # Check if the graph paused at an interrupt (e.g. ask_user tool)
    state = await graph.aget_state(config)
    all_interrupts = [i for task in state.tasks for i in task.interrupts]
    if all_interrupts:
        interrupt_val = all_interrupts[0].value
        question: str = (
            interrupt_val.get("question", "") if isinstance(interrupt_val, dict) else str(interrupt_val)
        )
        options: list[str] = (
            interrupt_val.get("options", []) if isinstance(interrupt_val, dict) else []
        )
        q_text = f"❓ <b>Agent 詢問：</b>\n{_h(question)}"
        if options:
            rows = [
                [InlineKeyboardButton(
                    opt,
                    callback_data={"a": "ask_reply", "tid": dag_thread_id, "iid": invoke_id, "ans": opt},
                )]
                for opt in options
            ]
            await bot.send_message(  # type: ignore[attr-defined]
                chat_id, text=q_text,
                message_thread_id=message_thread_id,
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(rows),
            )
        else:
            await bot.send_message(  # type: ignore[attr-defined]
                chat_id, text=q_text,
                message_thread_id=message_thread_id,
                parse_mode=ParseMode.HTML,
            )
        _pending_invoke[dag_thread_id] = invoke_id
        return None, None

    final_answer = answer_buf.strip()
    _, block_formulas = _extract_block_math(final_answer)
    final_text = (
        _build(_md_to_html(final_answer))
        if final_answer
        else _build("(無回應)")
    )
    final_msg: Message = await bot.send_message(  # type: ignore[attr-defined]
        chat_id,
        text=final_text,
        message_thread_id=message_thread_id,
        parse_mode=ParseMode.HTML,
    )
    for formula in block_formulas:
        try:
            buf = _render_formula_png(formula)
            await bot.send_photo(  # type: ignore[attr-defined]
                chat_id,
                photo=buf,
                message_thread_id=message_thread_id,
            )
        except Exception:
            logger.exception("Failed to render formula: %s", formula[:80])
    return final_msg, last_node_id


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

    dag_thread_id = _thread_id(message)

    # Cancel any in-flight debounce task for this thread
    existing = _pending_tasks.get(dag_thread_id)
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

        # /merge pending: pass parent IDs via configurable to graph.py
        pending_merge = _pending_merges.pop(dag_thread_id, None)

        # Resume a paused interrupt if one exists, otherwise start a new invoke
        pending_invoke_id = _pending_invoke.pop(dag_thread_id, None)
        if pending_invoke_id:
            invoke_id = pending_invoke_id
            resume_val: str | None = user_text
        else:
            invoke_id = generate_id()
            resume_val = None

        async with _thread_locks[dag_thread_id]:
            try:
                final_msg, node_id = await run_agent_streaming(
                    user_text,
                    dag_thread_id,
                    invoke_id,
                    context.bot,
                    message.chat_id,
                    message.message_thread_id,
                    resume_value=resume_val,
                    merge_parents=pending_merge,
                )
            except Exception as exc:
                logger.exception("Agent invocation failed for thread %s", dag_thread_id)
                await message.reply_text(f"⚠️ {_classify_error(exc)}")
                try:
                    await message.set_reaction([ReactionTypeEmoji("❌")])
                except Exception:
                    pass
                return

        if final_msg is None:
            # Agent interrupted — question already sent; show a thinking reaction
            try:
                await message.set_reaction([ReactionTypeEmoji("🤔")])
            except Exception:
                pass
            return

        # Attach inline keyboard (Regenerate / Branch here)
        kb = _make_reply_keyboard(dag_thread_id, user_text, node_id)
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
    _pending_tasks[dag_thread_id] = task


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
        try:
            await old_msg.delete()
        except Exception:
            pass
        invoke_id_regen = generate_id()
        async with _thread_locks[thread_id]:
            try:
                final_msg, regen_node_id = await run_agent_streaming(
                    user_message,
                    thread_id,
                    invoke_id_regen,
                    context.bot,
                    chat_id_regen,
                    thread_id_regen,
                )
            except Exception as exc:
                logger.exception("Regen failed for thread %s", thread_id)
                await context.bot.send_message(chat_id_regen, text=f"⚠️ {_classify_error(exc)}", message_thread_id=thread_id_regen)
                return
        if final_msg is not None:
            kb = _make_reply_keyboard(thread_id, user_message, regen_node_id)
            try:
                await final_msg.edit_reply_markup(reply_markup=kb)
            except Exception:
                pass

    elif action == "ask_reply":
        # Resume a paused ask_user interrupt from an inline button tap
        iid: str = data.get("iid", "")
        ans: str = data.get("ans", "")
        if not iid or not thread_id:
            return
        # Consume pending_invoke (button tap wins over any pending text)
        _pending_invoke.pop(thread_id, None)
        old_msg = query.message
        chat_id_ask: int = old_msg.chat_id
        thread_id_ask: int | None = old_msg.message_thread_id
        # Edit button message to show selected answer
        try:
            await old_msg.edit_text(
                old_msg.text_html + f"\n\n<b>✅ 選擇：</b>{_h(ans)}",
                parse_mode=ParseMode.HTML,
                reply_markup=None,
            )
        except Exception:
            pass
        async with _thread_locks[thread_id]:
            try:
                final_msg, ask_node_id = await run_agent_streaming(
                    ans,
                    thread_id,
                    iid,
                    context.bot,
                    chat_id_ask,
                    thread_id_ask,
                    resume_value=ans,
                )
            except Exception as exc:
                logger.exception("ask_reply resume failed for thread %s", thread_id)
                await context.bot.send_message(chat_id_ask, text=f"⚠️ {_classify_error(exc)}", message_thread_id=thread_id_ask)
                return
        if final_msg is not None:
            kb = _make_reply_keyboard(thread_id, ans, ask_node_id)
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
    """Handle /render -- render DAG topology to PNG and send as photo."""
    if update.message is None:
        return

    thread_id = _thread_id(update.message)
    jsonl_path = _jsonl_path(thread_id)

    if not jsonl_path.exists():
        await update.message.reply_text("No conversation history yet.")
        return

    dag_graph = load(jsonl_path)

    try:
        output_path = render_topology_png(dag_graph, jsonl_path)
    except Exception as exc:
        await update.message.reply_text(f"Render failed: {exc}")
        return

    with open(output_path, "rb") as f:
        await update.message.reply_photo(photo=f)


async def cmd_end_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /end_session -- manually flush session record to disk."""
    if update.message is None:
        return
    dag_thread_id = _thread_id(update.message)
    flushed = await flush_session(dag_thread_id)
    if flushed:
        await update.message.reply_text("✅ Session 已結束並寫入磁碟")
    else:
        await update.message.reply_text("無開放中的 session")


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
    """Register bot commands and start background tasks."""
    await app.bot.set_my_commands([
        BotCommand("branch",      "切換到指定節點（開始新分支）"),
        BotCommand("switch",      "/branch 別名"),
        BotCommand("list",        "列出最近的對話節點"),
        BotCommand("status",      "顯示目前會話狀態"),
        BotCommand("show",        "顯示當前節點到根的路徑"),
        BotCommand("paths",       "顯示所有分支路徑"),
        BotCommand("view",        "互動式節點瀏覽器"),
        BotCommand("merge",       "合併分支（選定父節點後發送問句）"),
        BotCommand("delete",      "軟刪除指定節點"),
        BotCommand("render",      "渲染 DAG 拓撲圖"),
        BotCommand("end_session", "立即結束並寫入當前 session"),
        BotCommand("maintain",    "整理 JSONL 索引"),
    ])
    start_session_flusher()
    me = await app.bot.get_me()
    print(f"Bot started: @{me.username}  |  logs → logs/bot.log")


async def _post_shutdown(app) -> None:  # type: ignore[type-arg]
    """Flush all open sessions on graceful shutdown."""
    await flush_all_sessions()
    logger.info("All sessions flushed on shutdown")


def main() -> None:
    """Start the Telegram bot."""
    setup_logging()
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
        .post_shutdown(_post_shutdown)
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
    app.add_handler(CommandHandler("render",      cmd_render))
    app.add_handler(CommandHandler("end_session", cmd_end_session))
    app.add_handler(CommandHandler("maintain",    cmd_maintain))

    # Inline keyboard callbacks (Regenerate / Branch here)
    app.add_handler(CallbackQueryHandler(cmd_callback))

    # Message handler (catch-all for non-command text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    app.run_polling()


if __name__ == "__main__":
    main()
