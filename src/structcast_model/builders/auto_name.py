"""Auto Name."""

from collections import defaultdict


class AutoName:
    """Auto Name."""

    def __init__(self, infix: str = "") -> None:
        """Initialize AutoName."""
        self._infix = infix
        self._object_name_uids: dict[str, int] = defaultdict(int)

    def __call__(self, value: str) -> str:
        """Generate a unique name."""
        if value in self._object_name_uids:
            unique_name = f"{value}{self._infix}{self._object_name_uids[value]}"
        else:
            unique_name = value
        self._object_name_uids[value] += 1
        return unique_name

    def reset(self) -> None:
        """Reset the name counter."""
        self._object_name_uids.clear()
