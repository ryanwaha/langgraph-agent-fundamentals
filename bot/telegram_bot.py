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
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, ReactionTypeEmoji, Update
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
from agent.graph import cancel_session_flusher, drain_compression_tasks, flush_all_sessions, flush_session, graph, start_session_flusher
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


# CJK font file candidates in priority order.
# WSL2 can read Windows fonts directly under /mnt/c/Windows/Fonts/.
_CJK_FONT_PATHS = [
    "/mnt/c/Windows/Fonts/msjh.ttc",          # 微軟正黑體 (Traditional Chinese)
    "/mnt/c/Windows/Fonts/NotoSansTC-VF.ttf",  # Noto Sans TC
    "/mnt/c/Windows/Fonts/kaiu.ttf",           # 標楷體
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux Noto
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
_cjk_font_cache: list[str | None] = [False]  # type: ignore[list-item]  # False = not yet probed


def _find_cjk_font() -> str | None:
    """Return path to the first available CJK font file, or None."""
    if _cjk_font_cache[0] is not False:
        return _cjk_font_cache[0]  # type: ignore[return-value]
    import os
    for path in _CJK_FONT_PATHS:
        if os.path.exists(path):
            logger.info("CJK font selected for formula rendering: %s", path)
            _cjk_font_cache[0] = path
            return path
    logger.warning("No CJK font found; \\text{} with Chinese will render as boxes")
    _cjk_font_cache[0] = None
    return None


def _render_formula_png(formula: str) -> io.BytesIO:
    """Render a LaTeX display-math formula to a PNG image using matplotlib.mathtext."""
    import logging as _logging
    import matplotlib  # lazy import — only needed when formulas are present
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # Load CJK font so \text{中文} inside mathtext renders correctly.
    # rcParams["font.family"] only affects regular text; mathtext \text{} uses
    # the "rm" slot from the mathtext font set, so we must use fontset="custom"
    # and point mathtext.rm at the CJK font.
    cjk_font_path = _find_cjk_font()
    if cjk_font_path:
        fm.fontManager.addfont(cjk_font_path)
        font_name = fm.FontProperties(fname=cjk_font_path).get_name()
        matplotlib.rcParams["mathtext.fontset"] = "custom"
        matplotlib.rcParams["mathtext.rm"] = font_name
        matplotlib.rcParams["mathtext.it"] = font_name
        matplotlib.rcParams["mathtext.bf"] = font_name

    # Silence matplotlib.mathtext WARNING/INFO that fire for every unrenderable
    # glyph — these are expected fallbacks, not actionable errors.
    _mpl_log = _logging.getLogger("matplotlib")
    _prev_level = _mpl_log.level
    _mpl_log.setLevel(_logging.ERROR)

    fig = plt.figure(figsize=(8, 1.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, f"${formula}$", size=18, ha="center", va="center",
            transform=ax.transAxes)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, pad_inches=0.3)
    plt.close(fig)
    _mpl_log.setLevel(_prev_level)
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


# Phase 1 (TTFT draft) limits
_P1_SOFT = 4000   # stop adding new content, start dot-padding
_P1_HARD = 4090   # hard truncate + warning

# Phase 2 (answer) split thresholds
_P2_SPLIT_NL2 = 3800   # look back for \n\n
_P2_SPLIT_NL  = 4000   # look back for \n
_P2_HARD      = 4090   # hard cut + warning


def _split_answer(text: str) -> list[str]:
    """Split answer text into Telegram-safe chunks with multi-level fallback."""
    if len(text) <= _P2_SPLIT_NL2:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= _P2_SPLIT_NL2:
            chunks.append(text)
            break
        cut = text.rfind("\n\n", 0, _P2_SPLIT_NL2)
        if cut != -1:
            chunks.append(text[:cut].rstrip())
            text = text[cut:].lstrip()
            continue
        cut = text.rfind("\n", 0, _P2_SPLIT_NL)
        if cut != -1:
            chunks.append(text[:cut].rstrip())
            text = text[cut:].lstrip()
            continue
        chunks.append(text[:_P2_HARD] + "\n⚠️ 已達字數上限")
        text = text[_P2_HARD:]
    return chunks


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
) -> tuple[list[int], str | None]:
    """Stream agent execution via sendMessageDraft.

    On normal completion returns (sent_message_ids, last_node_id).
    If the agent interrupts (e.g. ask_user tool), sends the question to the
    user and returns ([], None) — the invoke_id is stored in _pending_invoke
    so the next user reply resumes it.

    Phases:
      Phase 1 (TTFT): raw steps/thinking streamed via draft, no processing.
      Phase 2 (answer): steps compressed to blockquote, answer streamed via draft.
      Final: answer sent as real message(s), formula PNGs appended.
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

    # --- State ---
    steps: list[str] = []          # raw phase-1 lines; compressed in phase 2
    thinking_buf: str = ""
    thinking_step_idx: int = -1
    answer_buf: str = ""           # current answer segment
    phase: int = 1                 # 1 = TTFT, 2 = answer streaming
    p1_truncated: bool = False
    sent_messages: list[int] = []
    current_query: str | None = None
    last_node_id: str | None = None
    dot_task: asyncio.Task | None = None  # type: ignore[type-arg]

    # --- Draft debounce ---
    # send_message_draft is rate-limited. Buffer the latest text and flush at
    # most once per _DRAFT_INTERVAL seconds to stay under Telegram flood limits.
    _DRAFT_INTERVAL = 1.0
    _draft_buf: list[str] = [""]   # mutable cell: [latest_text]
    _draft_flush_task: asyncio.Task | None = None  # type: ignore[type-arg]
    _draft_flush_task_holder: list = [None]

    async def _flush_loop() -> None:
        while True:
            await asyncio.sleep(_DRAFT_INTERVAL)
            text = _draft_buf[0]
            if not text:
                continue
            try:
                await bot.send_message_draft(  # type: ignore[attr-defined]
                    chat_id,
                    draft_id=1,
                    text=text,
                    message_thread_id=message_thread_id,
                    parse_mode=ParseMode.HTML,
                )
                logger.debug("[push] draft flush  phase=%d  len=%d", phase, len(text))
            except Exception:
                logger.debug("[push] send_message_draft failed (suppressed)")

    _draft_flush_task_holder[0] = asyncio.create_task(_flush_loop())

    def _draft(text: str) -> None:
        """Queue text for the next draft flush (non-blocking)."""
        _draft_buf[0] = text

    async def _draft_stop() -> None:
        """Cancel flush loop and send one final draft."""
        t = _draft_flush_task_holder[0]
        if t and not t.done():
            t.cancel()
        _draft_buf[0] = ""

    def _p1_draft() -> str:
        # Display raw thinking content in Phase 1; steps[] keeps the status
        # line ("🧠 正在推理（N 字）...") for _compress_steps() to use later.
        display = list(steps)
        if thinking_buf and thinking_step_idx >= 0:
            display[thinking_step_idx] = _md_to_html(thinking_buf)
        text = "\n".join(display)
        if len(text) >= _P1_HARD:
            return text[:_P1_HARD - 12] + "\n⚠️ 已達預覽上限"
        return text

    async def _dot_pad() -> None:
        nonlocal p1_truncated
        logger.info("[push] dot_pad start  content_len=%d", len("\n".join(steps)))
        while True:
            await asyncio.sleep(1)
            content = "\n".join(steps)
            if len(content) >= _P1_HARD:
                p1_truncated = True
                logger.info("[push] dot_pad P1_HARD reached → truncated")
                _draft(_p1_draft())
                return
            if steps:
                steps[-1] = steps[-1] + "."
            _draft(_p1_draft())

    def _maybe_start_dot_task() -> asyncio.Task | None:  # type: ignore[type-arg]
        if not p1_truncated and len("\n".join(steps)) >= _P1_SOFT:
            logger.info("[push] dot_pad triggered  content_len=%d", len("\n".join(steps)))
            return asyncio.create_task(_dot_pad())
        return None

    def _compress_steps() -> str:
        """Compress raw steps into a single <blockquote expandable> for phase 2.

        Format (each search tool call):
          ✅ 已搜尋：「query」
            • <a href="url">url</a>
            • ...
        Followed by thinking summary (if any).
        """
        summary_lines: list[str] = []
        for s in steps:
            if not s.startswith("✅"):
                continue
            lines = s.split("\n")
            summary_lines.append(lines[0])          # ✅ 已搜尋：「...」
            for raw in lines[1:]:                   # "  • https://..."
                url = raw.strip().lstrip("• ").strip()
                if url:
                    summary_lines.append(f'  • <a href="{url}">{url}</a>')

        if thinking_buf:
            headings = re.findall(r"^#{1,6}\s+(.+)$", thinking_buf, re.MULTILINE)
            if headings:
                for h in headings:
                    summary_lines.append(f"🧠 {_h(h.strip())}")
                logger.debug("[push] compress_steps thinking  headings=%d", len(headings))
            else:
                # Fallback: first sentence of each \n\n paragraph
                sentences: list[str] = []
                for para in thinking_buf.split("\n\n"):
                    para = para.strip()
                    if not para:
                        continue
                    m = re.search(r"[^。.!?！？\n]+[。.!?！？]?", para)
                    if m:
                        sentences.append(m.group(0).strip())
                if sentences:
                    for s in sentences:
                        summary_lines.append(f"🧠 {_h(s)}")
                    logger.debug("[push] compress_steps thinking  sentences=%d", len(sentences))
                else:
                    summary_lines.append(f"🧠 已推理（{len(thinking_buf)} 字）")

        if not summary_lines:
            return ""
        inner = "\n".join(summary_lines)
        return f"<blockquote expandable>{inner}</blockquote>"

    def _p2_draft(answer_html: str) -> str:
        summary = _compress_steps()
        parts = [p for p in [summary, answer_html] if p]
        return "\n\n".join(parts)

    async def _send_segment(answer_html: str) -> None:
        """send_message the current blockquote+answer segment; append ids."""
        summary = _compress_steps()
        parts = [p for p in [summary, answer_html] if p]
        full = "\n\n".join(parts)
        chunks = _split_answer(full)
        logger.info("[push] send_segment  chunks=%d  total_len=%d  has_blockquote=%s",
                    len(chunks), len(full), bool(summary))
        for i, chunk in enumerate(chunks):
            msg = await bot.send_message(  # type: ignore[attr-defined]
                chat_id,
                text=chunk,
                message_thread_id=message_thread_id,
                parse_mode=ParseMode.HTML,
            )
            sent_messages.append(msg.message_id)
            logger.info("[push] sent chunk %d/%d  msg_id=%d  len=%d",
                        i + 1, len(chunks), msg.message_id, len(chunk))

    # --- Event loop ---

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
            logger.info("[push] model_start  phase=%d  steps=%d", phase, len(steps))
            _draft(_p1_draft() or "🤔 正在思考...")

        elif evt == "on_tool_start":
            tool_name: str = event["name"]
            tool_input: dict = event["data"].get("input") or {}
            label = _TOOL_LABELS.get(tool_name, tool_name)

            if phase == 2 and answer_buf.strip():
                # Finalize current answer segment before new tool call
                logger.info("[push] tool_start mid-answer  tool=%s  answer_len=%d → send_segment",
                            tool_name, len(answer_buf))
                if dot_task:
                    dot_task.cancel()
                await _send_segment(_md_to_html(answer_buf))
                answer_buf = ""
                steps.clear()

            if tool_name == "search":
                current_query = str(tool_input.get("query", ""))
                steps.append(f"⏳ 正在{label}：「{_h(current_query[:60])}」")
                logger.info("[push] tool_start  tool=search  query=%r  phase=%d", current_query[:60], phase)
            else:
                current_query = None
                steps.append(f"⏳ 正在執行：{_h(label)}")
                logger.info("[push] tool_start  tool=%s  phase=%d", tool_name, phase)

            if phase == 1:
                dot_task = _maybe_start_dot_task()
                _draft(_p1_draft())
            else:
                _draft(_p2_draft(""))

        elif evt == "on_tool_end":
            tool_name = event["name"]
            label = _TOOL_LABELS.get(tool_name, tool_name)
            tool_output = event["data"].get("output")

            for i in range(len(steps) - 1, -1, -1):
                if steps[i].startswith("⏳"):
                    if tool_name == "search" and current_query:
                        # Phase 1: include URLs; phase 2: _compress_steps() drops them
                        result_urls: list[str] = []
                        if isinstance(tool_output, list):
                            for r in tool_output:
                                url = r.get("url") if isinstance(r, dict) else None
                                if url:
                                    result_urls.append(url)
                        url_lines = "".join(f"\n  • {u}" for u in result_urls[:5])
                        steps[i] = f"✅ 已{label}：「{_h(current_query[:60])}」{url_lines}"
                        logger.info("[push] tool_end  tool=search  urls=%d  phase=%d",
                                    len(result_urls), phase)
                    else:
                        steps[i] = steps[i].replace("⏳ 正在", "✅ 已完成", 1)
                        logger.info("[push] tool_end  tool=%s  phase=%d", tool_name, phase)
                    break
            current_query = None

            if dot_task and not dot_task.done():
                dot_task.cancel()
                dot_task = None
            dot_task = _maybe_start_dot_task()

            if phase == 1:
                _draft(_p1_draft())
            else:
                _draft(_p2_draft(_md_to_html(answer_buf)))

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
                            logger.info("[push] thinking_start  phase=%d", phase)
                        thinking_buf += btext
                        if thinking_step_idx >= 0:
                            steps[thinking_step_idx] = f"🧠 正在推理（{len(thinking_buf)} 字）..."
                        if phase == 1:
                            if dot_task is None:
                                dot_task = _maybe_start_dot_task()
                            if not p1_truncated:
                                _draft(_p1_draft())
                    else:
                        if phase == 1:
                            # Transition to phase 2
                            if dot_task:
                                dot_task.cancel()
                                dot_task = None
                            # Finalize thinking step display
                            if thinking_step_idx >= 0:
                                thinking_step_idx = -1
                            phase = 2
                            logger.info("[push] phase 1→2  steps=%d  thinking_len=%d  p1_content_len=%d",
                                        len(steps), len(thinking_buf), len("\n".join(steps)))
                        answer_buf += btext
                        _draft(_p2_draft(_md_to_html(answer_buf)))

        elif evt == "on_chain_end" and node == "":
            output = event["data"].get("output") or {}
            last_node_id = output.get("last_node_id") or last_node_id

    # Cancel any lingering dot task
    if dot_task and not dot_task.done():
        dot_task.cancel()

    # Check if the graph paused at an interrupt (e.g. ask_user tool)
    state = await graph.aget_state(config)
    all_interrupts = [i for task in state.tasks for i in task.interrupts]
    if all_interrupts:
        logger.info("[push] interrupt detected  count=%d", len(all_interrupts))
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
        await _draft_stop()
        _pending_invoke[dag_thread_id] = invoke_id
        return [], None

    await _draft_stop()

    # Send final answer segment
    final_answer = answer_buf.strip()
    _, block_formulas = _extract_block_math(final_answer)
    answer_html = _md_to_html(final_answer) if final_answer else "(無回應)"
    await _send_segment(answer_html)

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

    logger.info("[push] done  sent=%s  last_node=%s", sent_messages, last_node_id)
    return sent_messages, last_node_id


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
        # React 👀 immediately to acknowledge receipt
        try:
            await message.set_reaction([ReactionTypeEmoji("👀")])
        except Exception:
            pass

        # Debounce window: if another message arrives within 800 ms, this task
        # will be cancelled before the sleep completes.
        await asyncio.sleep(0.8)

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
                msg_ids, node_id = await run_agent_streaming(
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

        if not msg_ids:
            # Agent interrupted — question already sent; show a thinking reaction
            try:
                await message.set_reaction([ReactionTypeEmoji("🤔")])
            except Exception:
                pass
            return

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
    """Handle inline keyboard button presses."""
    query = update.callback_query
    if query is None:
        return
    await query.answer()  # dismiss Telegram spinner

    data: dict = query.data  # arbitrary_callback_data delivers the original dict
    action = data.get("a")
    thread_id: str = data.get("tid", "")

    if action == "ask_reply":
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
                await run_agent_streaming(
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
    cancel_session_flusher()
    await drain_compression_tasks(timeout=30.0)  # wait for in-flight compressions before session flush
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
