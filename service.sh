#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
python3 settings.py >/dev/null
COMPOSE=(docker compose --env-file "$ROOT/.env" -f "$ROOT/compose.yaml")
approve() { [[ "${2:-}" == --approved ]] || { echo "Use $1 --approved for service-changing operations" >&2; exit 1; }; }
case "${1:-status}" in
  check) python3 check.py "${@:2}" ;;
  prepare) python3 prepare.py ;;
  start) approve "$@"; python3 check.py; "${COMPOSE[@]}" up -d --pull never ;;
  restart)
    approve "$@"
    python3 check.py --config-only
    # Only this Compose project; no other model service is stopped.
    "${COMPOSE[@]}" stop --timeout 60
    python3 check.py
    "${COMPOSE[@]}" up -d --pull never
    ;;
  stop) approve "$@"; "${COMPOSE[@]}" stop --timeout 60 ;;
  status) "${COMPOSE[@]}" ps ;;
  logs) "${COMPOSE[@]}" logs "${@:2}" model ;;
  verify) python3 verify.py ;;
  *) echo "Usage: $0 {prepare|check [--config-only]|start --approved|restart --approved|stop --approved|status|logs|verify}" >&2; exit 1 ;;
esac
