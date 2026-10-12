import pytest

from kinveil.tools import Tool, ToolRegistry


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register(Tool("clock", "Read the time", lambda: "now"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Tool("clock", "Another clock", lambda: "later"))
