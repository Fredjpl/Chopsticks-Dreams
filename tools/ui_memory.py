# tools/ui_memory.py
class ConversationMemory:
    """Stores conversation history for multi-turn context."""
    def __init__(self, max_history=5):
        self.max_history = max_history  # number of past Q&A pairs to retain
        self.history = []  # list of (user_message, bot_reply)
        self.last_rag: str | None = None   # cache of last RAG passages
    
    def add_interaction(self, user_message: str, bot_reply: str):
        """Add a user question and bot answer to history."""
        self.history.append((user_message, bot_reply))
        # Trim history to max_length
        if len(self.history) > self.max_history:
            self.history.pop(0)

    # ─── debugging helper ─────────────────────────────────────────────────
    def save_to_disk(self, file_path: str) -> None:
        """Write the current history to a plain‑text file (overwrites)."""
        with open(file_path, "w", encoding="utf‑8") as f:
            for u, b in self.history:
                f.write(f"User: {u}\n")
                f.write(f"ChefBot: {b}\n\n")
    
    def get_context(self) -> str:
        """Get formatted context from recent history for prompting."""
        if not self.history:
            return ""
        context_lines = []
        for u_msg, b_msg in self.history:
            context_lines.append(f"User asked: {u_msg}")
            context_lines.append(f"ChefBot answered: {b_msg}")
            context_lines.append("")  # blank line between interactions
        return "\n".join(context_lines)
