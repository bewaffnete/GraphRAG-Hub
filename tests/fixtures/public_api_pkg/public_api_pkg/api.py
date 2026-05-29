class PublicClient:
    """Public API client."""

    def run(self, value: int) -> str:
        """Run the public operation.

        Example:
            >>> client = PublicClient()
            >>> client.run(3)
            '3'
        """
        if value < 0:
            raise ValueError("value must be non-negative")
        return str(value)

    def _hidden_method(self) -> None:
        return None


def public_helper(name: str) -> str:
    """Build a public greeting."""
    return name.upper()


def _private_helper() -> str:
    return "hidden"
