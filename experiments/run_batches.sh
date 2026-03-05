#!/usr/bin/env bash
set +e  # do NOT exit on error

LIST_FILE="experiments/list.sh"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

if [[ ! -f "$LIST_FILE" ]]; then
  echo "ERROR: list file not found: $LIST_FILE" >&2
  exit 1
fi

while IFS= read -r f || [[ -n "$f" ]]; do
  # skip empty lines and comments
  [[ -z "$f" ]] && continue
  [[ "$f" =~ ^[[:space:]]*# ]] && continue

  echo "=== Running: $f ==="

  if [[ ! -f "$f" ]]; then
    echo "MISSING: $f (skipping)"
    echo "$f | MISSING" >> "$LOG_DIR/failed.txt"
    continue
  fi

  base="$(basename "$f")"
  log="$LOG_DIR/${base%.sh}.log"

  bash "$f" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}

  if [[ $status -ne 0 ]]; then
    echo "FAILED ($status): $f (see $log)"
    echo "$f | $status" >> "$LOG_DIR/failed.txt"
  else
    echo "OK: $f"
  fi
done < "$LIST_FILE"

echo "=== Done. Failures (if any) are in $LOG_DIR/failed.txt ==="
exit 0