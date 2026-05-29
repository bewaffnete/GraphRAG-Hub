"""Context compression service."""


class ContextCompressor:
    def compress(self, items: list[str], max_items: int = 5) -> str:
        visible = [item.strip() for item in items if item.strip()]
        return " | ".join(visible[:max_items])
