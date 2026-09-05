# syntax=docker/dockerfile:1

FROM golang:1.25-bookworm AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/autogravity ./cmd/autogravity

FROM debian:bookworm-slim AS assets
ARG TARGETARCH
ARG ONNXRUNTIME_VERSION=1.23.2
ARG MODEL_URL=https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx
ARG MODEL_SHA256=309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && case "${TARGETARCH}" in \
         amd64) ORT_ARCH=x64 ;; \
         arm64) ORT_ARCH=aarch64 ;; \
         *) echo "unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
       esac \
    && curl -fL --retry 3 -o /tmp/onnxruntime.tgz \
       "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ORT_ARCH}-${ONNXRUNTIME_VERSION}.tgz" \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/onnxruntime.tgz --strip-components=1 -C /opt/onnxruntime \
    && mkdir -p /opt/models \
    && curl -fL --retry 3 -o /opt/models/u2netp.onnx "${MODEL_URL}" \
    && echo "${MODEL_SHA256}  /opt/models/u2netp.onnx" | sha256sum -c - \
    && rm /tmp/onnxruntime.tgz

FROM debian:bookworm-slim
ARG ONNXRUNTIME_VERSION=1.23.2
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system autogravity \
    && useradd --system --gid autogravity --home-dir /nonexistent autogravity
COPY --from=builder /out/autogravity /usr/local/bin/autogravity
COPY --from=assets /opt/onnxruntime/lib /opt/onnxruntime/lib
COPY --from=assets /opt/models /opt/models
ENV ADDR=:8080 \
    MODEL_PATH=/opt/models/u2netp.onnx \
    ONNXRUNTIME_LIB=/opt/onnxruntime/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION}
USER autogravity
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/autogravity"]
