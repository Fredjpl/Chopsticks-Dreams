"""tools/ui.py: Terminal UI for Chinese Recipe RAG Chatbot.

This module manages the interactive conversation loop and printing of messages
with appropriate formatting (colors, prefixes, spacing). It differentiates 
User vs. ChefBot messages, handles long texts gracefully, and pauses after 
each bot reply for better readability.
"""
import sys
import os
import asyncio
from tools.ui_memory import ConversationMemory

# Optional color support (use colorama if available for Windows, otherwise ANSI)
COLOR_ENABLED = False
try:
    import colorama
    colorama.init()  # initialize colorama for Windows ANSI support
    from colorama import Fore, Style
    COLOR_ENABLED = True
except ImportError:
    # colorama not installed; use ANSI codes if terminal supports, else no color
    if sys.stdout.isatty() and os.environ.get("TERM") != "dumb":
        # Define ANSI escape sequences for colors
        class ANSIColor:
            """Simple container for ANSI color codes."""
            pass
        Fore = ANSIColor()  # dummy class to hold attributes
        Style = ANSIColor()
        Fore.CYAN = "\033[96m"   # Bright Cyan
        Fore.GREEN = "\033[92m"  # Bright Green
        Style.RESET_ALL = "\033[0m"
        COLOR_ENABLED = True
    else:
        # No color support
        class NoColor:
            """Fallback color class with empty codes."""
            def __getattr__(self, item):
                return ""  # Any attribute returns empty string
        Fore = NoColor()
        Style = NoColor()
        # COLOR_ENABLED remains False (no colors will be used)

# Define colored prefixes for User and ChefBot
USER_PREFIX = f"{Fore.CYAN}User: {Style.RESET_ALL}" if COLOR_ENABLED else "User: "
BOT_PREFIX  = f"{Fore.GREEN}ChefBot: {Style.RESET_ALL}" if COLOR_ENABLED else "ChefBot: "

def format_user_message(message: str) -> str:
    """Format a user's message with the User prefix and color."""
    return f"{Fore.CYAN if COLOR_ENABLED else ''}User: {message}{Style.RESET_ALL if COLOR_ENABLED else ''}"

def format_bot_message(message: str) -> str:
    """Format the bot's message with the ChefBot prefix and color."""
    return f"{Fore.GREEN if COLOR_ENABLED else ''}ChefBot: {message}{Style.RESET_ALL if COLOR_ENABLED else ''}"

def display_user_message(message: str) -> None:
    """Print the user message to the console with proper format and spacing."""
    print(format_user_message(message))
    print()  # blank line for spacing after user message

def display_bot_message(message: str) -> None:
    """Print the bot message to the console with proper format and spacing."""
    print(format_bot_message(message))
    print()  # blank line for spacing after bot message

def prompt_user(prompt: str = "User: ") -> str:
    """
    Prompt the user for input, using a colored prefix if available.
    Returns the user's input string.
    """
    if COLOR_ENABLED:
        # Show the prompt in cyan, then reset so user input is not colored.
        prompt_str = Fore.CYAN + prompt + Style.RESET_ALL
    else:
        prompt_str = prompt
    try:
        return input(prompt_str)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully to exit the chat.
        print("\n[Session terminated by user]")
        sys.exit(0)

async def chat_loop(get_response):
    """
    Run the main chat loop, continuously reading user input and printing bot responses.
    `get_response` should be a function or coroutine that takes a user query string 
    and returns the bot's reply string.
    """
    # Welcome message
    print("Welcome to ChefBot! Ask me anything about Chinese recipes.\n")
    memory = ConversationMemory(max_history=5) 
    # Loop for user queries
    while True:
        # 1. Get raw user input
        user_query = prompt_user()
        if user_query is None:
            continue  # just in case, though input() returns "" not None on Enter
        user_query = user_query.strip()

        # Check exit conditions
        if user_query.lower() in {"exit", "quit", "q"}:
            print("Goodbye! 👋")
            break

        # 2. Display the user's message in the conversation log
        display_user_message(user_query)

        # 3. Grab the last 3 Q‑A pairs as structured history
        history_pairs = memory.history[-5:]
        try:
            if asyncio.iscoroutinefunction(get_response):
                bot_reply = await get_response(user_query, history_pairs)
            else:
                bot_reply = await asyncio.get_event_loop().run_in_executor(
                    None, get_response, user_query, history_pairs
                )
        except Exception as e:
            # Handle any errors during retrieval/response generation
            error_msg = f"[Error] {e}"
            # Print error in red if available, otherwise default color
            if COLOR_ENABLED:
                print(f"{error_msg}{Style.RESET_ALL}\n")
            else:
                print(error_msg + "\n")
            # Continue to next iteration (prompt user again)
            continue

        # 4. Store + display the bot’s response
        memory.add_interaction(user_query, bot_reply)
        display_bot_message(bot_reply)

        # 5. Pause for the user to read the answer, before next prompt
        try:
            input("Press Enter to continue...")
        except KeyboardInterrupt:
            print("\n[Session terminated by user]")
            break

# If this module is run directly, we might want to start the chat with a default backend.
if __name__ == "__main__":
    try:
        # Import the recipe RAG answer generator (to integrate with chat loop)
        from tools import chef_agent
    except ImportError:
        chef_agent = None

    if chef_agent and hasattr(chef_agent, "answer_query"):
        # Use the answer_query function from recipe_rag if available
        asyncio.run(chat_loop(chef_agent.answer_query))
    else:
        # Fallback: echo mode or error if recipe_rag not available
        async def echo_response(msg):
            return "Echo: " + msg
        asyncio.run(chat_loop(echo_response))
