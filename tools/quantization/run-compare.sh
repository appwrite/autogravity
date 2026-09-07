#!/bin/sh
# Old workspace supplies the compiled Go service, runtime models, and Python venv.
set -eu
old_root=${1:?original workspace required}
expanded_root=${2:?expanded workspace required}
containers=
cleanup() {
    for id in $containers; do
        docker stop --time 30 "$id" >/dev/null 2>&1 || true
        docker rm "$id" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM
for model in fp32 previous candidate; do
    case "$model" in
        fp32) path=/work/u2net.onnx ;;
        previous) path=/work/u8s8-reduced.onnx ;;
        candidate) path=/expanded/u8s8-reduced.onnx ;;
    esac
    id=$(docker run -d --label autogravity.experiment=expanded --cpus=4 --memory=4g \
        -p 127.0.0.1::8080 -e GOMAXPROCS=4 -e MAX_CONCURRENT_ANALYSES=4 \
        -e ONNX_INTRA_OP_THREADS=1 -e MODEL_PATH="$path" \
        -v "$old_root:/work:ro" -v "$expanded_root:/expanded:ro" \
        --entrypoint /work/autogravity ghcr.io/appwrite/autogravity:0.0.5)
    containers="$containers $id"
    port=$(docker port "$id" 8080/tcp)
    url="http://$port"
    case "$model" in
        fp32) baseline=$url ;;
        previous) previous=$url ;;
        candidate) candidate=$url ;;
    esac
done
# Only four outstanding inference requests across all three services. The client
# has a separate one-CPU cap; this evaluation is not a timing comparison.
docker run --rm --network host --cpus=1 --memory=1g \
    -v "$old_root:/work:ro" -v "$expanded_root:/expanded" \
    python:3.11-slim /work/venv/bin/python /expanded/compare_http.py \
    --root /expanded --baseline "$baseline" --previous "$previous" --candidate "$candidate" --workers 4
