#!/usr/bin/env python3
import json
import sys

data = json.load(sys.stdin)
prompt = data.get("prompt", "").lower()

reminders = {
    "install": "Use 'uv add <package>', not pip install.",
    "test": "Run tests with 'uv run pytest'.",
    "format": "Format with 'uv run ruff format .'.",
    "lint": "Lint with 'uv run ruff check .'.",
    "type": "Type-check with 'uv run ty check'.",
    "run": "Run scripts with 'uv run <script>.py'.",
}

matched = [msg for kw, msg in reminders.items() if kw in prompt]
if matched:
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": " ".join(matched),
        }
    }
    json.dump(output, sys.stdout)
