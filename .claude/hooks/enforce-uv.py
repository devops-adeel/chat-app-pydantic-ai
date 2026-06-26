#!/usr/bin/env python3
import json
import sys

data = json.load(sys.stdin)
command = data.get("tool_input", {}).get("command", "")

redirects = {
    "pip install": "uv add",
    "pip3 install": "uv add",
    "pip uninstall": "uv remove",
    "pip3 uninstall": "uv remove",
    "python -m pip": "uv add / uv remove",
    "python3 -m pip": "uv add / uv remove",
    "python -m pytest": "uv run pytest",
    "python3 -m pytest": "uv run pytest",
    "python -m ruff": "uv run ruff",
    "python3 -m ruff": "uv run ruff",
}

for pattern, replacement in redirects.items():
    if pattern in command:
        print(f"Blocked: '{pattern}'. Use '{replacement}' instead.", file=sys.stderr)
        sys.exit(2)

bare = {"python ", "python3 "}
for b in bare:
    if command.startswith(b) and not command.startswith("uv run"):
        print(f"Blocked: use 'uv run {command}' instead.", file=sys.stderr)
        sys.exit(2)
