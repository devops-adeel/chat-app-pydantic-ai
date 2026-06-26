#!/usr/bin/env bash
input=$(cat)
file_path=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

if [[ "$file_path" == *.py ]] && [[ -f "$file_path" ]]; then
  "$HOME/.local/bin/uv" run ruff check --fix --quiet "$file_path" 2>/dev/null
  "$HOME/.local/bin/uv" run ruff format --quiet "$file_path" 2>/dev/null
fi
exit 0
