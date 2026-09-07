#!/bin/sh
# Run from the repository root. Requires Docker, curl, and Python 3.
set -eu
image=${1:?image tag required}
container=
cleanup() {
    if [ -n "$container" ]; then
        docker stop --time 10 "$container" >/dev/null 2>&1 || true
        docker rm "$container" >/dev/null 2>&1 || true
        container=
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM

for mode in default fp32 custom; do
    case "$mode" in
        default) set --; expected=models/u2net-int8.onnx ;;
        fp32) set -- -e MODEL_PRECISION=fp32; expected=models/u2net.onnx ;;
        custom) set -- -e MODEL_PRECISION=invalid -e MODEL_PATH=/opt/models/u2net.onnx; expected=/opt/models/u2net.onnx ;;
    esac
    container=$(docker run -d --cpus=2 --memory=4g -p 127.0.0.1::8080 "$@" "$image")
    address=$(docker port "$container" 8080/tcp)
    url="http://$address"
    ready=false
    for attempt in $(seq 1 60); do
        if curl -fsS --max-time 2 "$url/healthz" >/dev/null 2>&1; then
            ready=true
            break
        fi
        sleep 1
    done
    if [ "$ready" != true ]; then
        docker logs "$container"
        exit 1
    fi
    logs=$(docker logs "$container" 2>&1)
    case "$logs" in
        *"path=$expected "*) ;;
        *) echo "Expected selected model $expected; logs: $logs" >&2; exit 1 ;;
    esac
    curl -fsS --max-time 60 -H 'Content-Type: image/jpeg' \
        --data-binary @internal/testimages/testdata/dog-portrait.jpg "$url/analyze" \
        | python3 -c 'import json,sys; r=json.load(sys.stdin); assert .25 <= r["gravity"]["x"] <= .75; assert .58 <= r["gravity"]["y"] <= .85; assert 0 <= r["confidence"] <= 1; print(r)'
    echo "Container mode $mode passed"
    cleanup
done

if output=$(docker run --rm --cpus=2 --memory=4g -e MODEL_PRECISION=invalid "$image" 2>&1); then
    echo "Invalid precision unexpectedly succeeded" >&2
    exit 1
fi
case "$output" in
    *"MODEL_PRECISION must be int8 or fp32"*) ;;
    *) echo "Unexpected startup failure: $output" >&2; exit 1 ;;
esac
echo "Invalid precision rejected"
