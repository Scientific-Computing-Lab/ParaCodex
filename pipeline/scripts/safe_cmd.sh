#!/usr/bin/env bash
set -euo pipefail

OUT=""
MAX_LINES=80
MODE="tail"
GREP_PATTERN=""

usage() {
  cat <<'EOF'
Usage: safe_cmd.sh --out <file> [--max-lines N] [--mode head|tail|grep] [--grep <pattern>] -- <command...>
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT="${2:-}"; shift 2 ;;
    --max-lines)
      MAX_LINES="${2:-}"; shift 2 ;;
    --mode)
      MODE="${2:-}"; shift 2 ;;
    --grep)
      GREP_PATTERN="${2:-}"; shift 2 ;;
    --)
      shift; break ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage; exit 2 ;;
  esac
done

if [[ -z "${OUT}" ]]; then
  echo "--out is required" >&2
  usage
  exit 2
fi

if [[ $# -lt 1 ]]; then
  echo "Command is required" >&2
  usage
  exit 2
fi

mkdir -p "$(dirname "$OUT")"

set +e
"$@" >"$OUT" 2>&1
RC=$?
set -e

LINE_COUNT=$(wc -l < "$OUT" | tr -d ' ')
echo "Exit code: $RC"
echo "Output file: $OUT"
echo "Total lines: $LINE_COUNT"

if [[ "$LINE_COUNT" -eq 0 ]]; then
  exit "$RC"
fi

case "$MODE" in
  head)
    head -n "$MAX_LINES" "$OUT"
    ;;
  tail)
    tail -n "$MAX_LINES" "$OUT"
    ;;
  grep)
    if [[ -z "$GREP_PATTERN" ]]; then
      echo "Missing --grep pattern for grep mode" >&2
      exit 2
    fi
    grep -E "$GREP_PATTERN" "$OUT" | head -n "$MAX_LINES" || true
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac

exit "$RC"
