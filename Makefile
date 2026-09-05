MODEL_PATH := models/u2netp.onnx
MODEL_URL := https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx
MODEL_SHA256 := 309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8

.PHONY: build run test model

build:
	go build -o autogravity ./cmd/autogravity

run: model
	go run ./cmd/autogravity

test:
	go test ./...

model:
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		curl -fL --retry 3 -o "$(MODEL_PATH)" "$(MODEL_URL)"; \
	fi
	@echo "$(MODEL_SHA256)  $(MODEL_PATH)" | shasum -a 256 -c
