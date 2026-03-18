"""DAG engine import shim and helpers.

Adds the jsonl-dag-engine submodule to sys.path and re-exports its public API.
The helper ``build_dag_context`` constructs the conversation + narrative context
string for injection into the LangGraph agent's system message.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add jsonl-dag-engine to sys.path so its bare imports resolve.
_ENGINE_DIR = str(Path(__file__).resolve().parent.parent / "jsonl-dag-engine")
if _ENGINE_DIR not in sys.path:
    sys.path.insert(0, _ENGINE_DIR)

from dag_engine import (  # noqa: E402
    Graph,
    Node,
    load,
    append_node,
    write_session,
    switch_active,
    init_jsonl,
    generate_id,
    delete_node,
    render_topology_svg,
    maintain,
)
from prompt_builder import PromptBuilder  # noqa: E402

__all__ = [
    "Graph",
    "Node",
    "load",
    "append_node",
    "write_session",
    "switch_active",
    "init_jsonl",
    "generate_id",
    "delete_node",
    "render_topology_svg",
    "maintain",
    "PromptBuilder",
    "build_dag_context",
]


def build_dag_context(dag_graph: Graph) -> str:
    """Build the DAG conversation + narrative context string.

    Uses PromptBuilder's internal rendering to produce ``<conversation>``
    and ``<narrative>`` XML blocks, WITHOUT system_instructions or
    ``<user_query>``.  This output is appended to the agent's system message.

    Returns empty string when the graph has no active node (first message).
    """
    if dag_graph.active_node is None:
        return ""

    nodes = dag_graph.path_to_root(dag_graph.active_node)
    if not nodes:
        return ""

    builder = PromptBuilder(
        include_summary=False,
        include_time_flow=True,
        prompt_dump_path=None,
    )

    sections: list[str] = []
    sections.append(builder._render_conversation(nodes, graph=dag_graph))
    sections.append(builder._render_time_flow(dag_graph))
    return "\n\n".join(sections)
