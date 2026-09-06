# syntax=docker/dockerfile:1

FROM scratch AS onnxruntime-amd64
ADD --checksum=sha256:1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b \
    https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-x64-1.23.2.tgz /onnxruntime.tgz

FROM scratch AS onnxruntime-arm64
ADD --checksum=sha256:7c63c73560ed76b1fac6cff8204ffe34fe180e70d6582b5332ec094810241e5c \
    https://github.com/microsoft/onnxruntime/releases/download/v1.23.2/onnxruntime-linux-aarch64-1.23.2.tgz /onnxruntime.tgz

FROM onnxruntime-${TARGETARCH} AS onnxruntime

FROM golang:1.25-bookworm AS builder
COPY --from=onnxruntime /onnxruntime.tgz /tmp/onnxruntime.tgz
RUN mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/onnxruntime.tgz --strip-components=1 -C /opt/onnxruntime
ADD --checksum=sha256:309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8 --chmod=0444 \
    https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx /opt/models/u2netp.onnx

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/autogravity ./cmd/autogravity

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /out/autogravity /usr/local/bin/autogravity
COPY --from=builder /opt/onnxruntime/lib /opt/onnxruntime/lib
COPY --from=builder /opt/models /opt/models
ENV ADDR=:8080 \
    MODEL_PATH=/opt/models/u2netp.onnx \
    ONNXRUNTIME_LIB=/opt/onnxruntime/lib/libonnxruntime.so.1.23.2
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/autogravity"]
