"""Tool registry for the agent.

To add a new tool:
    1. Define an async function with a descriptive docstring (used as the tool description).
    2. Decorate it with @register_tool.
    That's it — it will be available to the agent automatically.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch
from langgraph.runtime import get_runtime

from agent.context import Context

# Central tool registry — all tools the agent can use
TOOLS: List[Callable[..., Any]] = []


def register_tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that registers a function as an agent tool."""
    TOOLS.append(func)
    return func


@register_tool
async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    Useful for answering questions about current events, facts, or anything
    requiring up-to-date information from the web.
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))
