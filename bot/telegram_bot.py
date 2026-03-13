"""Telegram bot entry point.

Bridges Telegram messages to the LangGraph ReAct agent.
Each chat gets its own thread_id, enabling per-chat conversation
history once a checkpointer is added to the graph.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from telegram import Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from agent.context import Context
from agent.graph import graph
from agent.state import InputState
from agent.utils import get_message_text

load_dotenv()


# --- Graph invocation ---

async def run_agent(user_message: str, thread_id: str) -> str:
    """Invoke the LangGraph agent and return the text response.

    Args:
        user_message: The raw text from the Telegram user.
        thread_id: Unique identifier for this conversation thread (chat ID).

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


# --- Telegram event handlers ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming Telegram messages.

    In private chats, always responds.
    In group chats, only responds when @mentioned.
    Uses the chat ID as the thread_id for conversation continuity.
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
        # Strip the mention from the message text
        user_text = text.replace(mention, "").strip()
    else:
        # Private chat — always respond
        user_text = text.strip()

    if not user_text:
        await message.reply_text("How can I help you? Ask me anything!")
        return

    thread_id = str(message.chat_id)

    # Send typing indicator (lasts ~5 seconds)
    await message.reply_chat_action(ChatAction.TYPING)

    try:
        response = await run_agent(user_text, thread_id)
    except Exception as exc:
        response = f"Sorry, I encountered an error: {exc}"

    # Telegram hard limit is 4096 chars; truncate with notice if needed
    if len(response) > 4096:
        response = response[:4093] + "..."

    await message.reply_text(response)


# --- Entry point ---

def main() -> None:
    """Start the Telegram bot."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment or .env file")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
