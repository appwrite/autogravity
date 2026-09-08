#!/bin/sh
# Docker HEALTHCHECK probe for autogravity.
# Derives the probe target from the runtime ADDR value (host:port or :port)
# so the check stays valid when ADDR is overridden at container runtime.
set -eu

addr="${ADDR:-:8080}"
port="${addr##*:}"
host="${addr%:*}"
case "$host" in
	"" | "0.0.0.0" | "[::]" | "*") host="127.0.0.1" ;;
esac

exec curl -f "http://$host:${port:-8080}/readyz"
