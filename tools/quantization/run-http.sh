#!/bin/sh
# Run on the x86 host after exporting tensors, evaluating the models, and
# building /work/autogravity. The image supplies only the pinned native runtime.
set -eu
work_root=${1:?absolute experiment workspace required}
candidate=${2:?candidate model name required}
duration=${3:-30}
cpu_counts=${4:-"1 2 4 6"}
container=

cleanup() {
    if [ -n "$container" ]; then
        docker stop --time 60 "$container" >/dev/null 2>&1 || true
        docker rm "$container" >/dev/null 2>&1 || true
        container=
    fi
}

trap cleanup EXIT HUP INT TERM

for workers in $cpu_counts; do
    case "$workers" in
        1|2|3|4|5|6|7|8) ;;
        *) echo "CPU counts must be integers from 1 to 8" >&2; exit 1 ;;
    esac
    for variant in u2net "$candidate"; do
        container=$(docker run -d --label autogravity.experiment=quantization \
            --cpus="$workers" --memory=4g \
            -p 127.0.0.1::8080 \
            -e GOMAXPROCS="$workers" -e MAX_CONCURRENT_ANALYSES="$workers" \
            -e ONNX_INTRA_OP_THREADS=1 -e MODEL_PATH="/work/$variant.onnx" \
            -v "$work_root:/work:ro" --entrypoint /work/autogravity \
            ghcr.io/appwrite/autogravity:0.0.5)
        # Save complete benchmark output before appending it to the matrix.
        python3 "$work_root/http_bench.py" "$container" "$variant" \
            --root "$work_root" --workers "$workers" --seconds "$duration" \
            > "$work_root/http-$variant-$workers.json"
        cat "$work_root/http-$variant-$workers.json"
        cleanup
    done
done
