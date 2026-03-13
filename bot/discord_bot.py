"""Discord bot entry point.

Bridges Discord messages to the LangGraph ReAct agent.
Each channel gets its own thread_id, enabling per-channel conversation
history once a checkpointer is added to the graph.
"""

import asyncio
import os

import discord
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.context import Context
from agent.graph import graph
from agent.state import InputState
from agent.utils import get_message_text

load_dotenv()

# --- Discord client setup ---

intents = discord.Intents.default()
intents.message_content = True  # Required privileged intent for reading message text
bot = discord.Client(intents=intents)


# --- Graph invocation ---

async def run_agent(user_message: str, thread_id: str) -> str:
    """Invoke the LangGraph agent and return the text response.

    Args:
        user_message: The raw text from the Discord user.
        thread_id: Unique identifier for this conversation thread (channel ID).

    Returns:
        The agent's final text response.
    """
    config = RunnableConfig(
        configurable={
            "thread_id": thread_id,
            # Add checkpointer to graph compile() to enable conversation memory
        }
    )

    result = await graph.ainvoke(
        InputState(messages=[HumanMessage(content=user_message)]),
        config=config,
        context=Context(),
    )

    final_message = result["messages"][-1]
    return get_message_text(final_message)


# --- Discord event handlers ---

@bot.event
async def on_ready() -> None:
    """Log successful connection."""
    assert bot.user is not None
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("------")


@bot.event
async def on_message(message: discord.Message) -> None:
    """Handle incoming Discord messages.

    Only responds when the bot is mentioned directly.
    Uses the channel ID as the thread_id for conversation continuity.
    """
    # Never respond to self
    if bot.user is None or message.author == bot.user:
        return

    # Only respond when explicitly mentioned
    if bot.user not in message.mentions:
        return

    # Strip the mention from the message text
    user_text = message.content.replace(f"<@{bot.user.id}>", "").strip()

    if not user_text:
        await message.reply("How can I help you? Ask me anything!")
        return

    thread_id = str(message.channel.id)

    async with message.channel.typing():
        try:
            response = await run_agent(user_text, thread_id)
        except Exception as exc:
            response = f"Sorry, I encountered an error: {exc}"

    # Discord hard limit is 2000 chars; truncate with notice if needed
    if len(response) > 1900:
        response = response[:1897] + "..."

    await message.reply(response)


# --- Entry point ---

def main() -> None:
    """Start the Discord bot."""
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN not set in environment or .env file")
    bot.run(token)


if __name__ == "__main__":
    main()
