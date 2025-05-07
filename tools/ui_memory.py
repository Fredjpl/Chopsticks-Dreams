class ConversationMemory:
    """Stores conversation history, last RAG context, and the current topic."""
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.history: list[tuple[str, str]] = []  # (user, bot)
        self.last_rag:   str | None = None        # cached RAG passages
        self.last_topic: str | None = None        # last detected dish / ingredient set

    # ─────────────────────────────── storage ──────────────────────────────
    def add_interaction(self, user_msg: str, bot_msg: str) -> None:
        self.history.append((user_msg, bot_msg))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    # ────────────────────────────── utilities ─────────────────────────────
    def save_to_disk(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for u, b in self.history:
                f.write(f"User: {u}\nChefBot: {b}\n\n")

    def get_context(self) -> str:
        lines: list[str] = []
        for u, b in self.history:
            lines += [f"User asked: {u}", f"ChefBot answered: {b}", ""]
        return "\n".join(lines)

