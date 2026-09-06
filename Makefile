MODEL_PATH := models/u2net.onnx
MODEL_URL := https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
MODEL_SHA256 := 8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491

.PHONY: build run test test-integration evaluate model

build:
	go build -o autogravity ./cmd/autogravity

run: model
	go run ./cmd/autogravity

test:
	go test ./...

test-integration: model
	go test -race -tags=integration ./...

evaluate: model
	go test -tags=integration,evaluation -run TestEvaluateDifficultScenes -v ./cmd/autogravity

model:
	@if [ ! -f "$(MODEL_PATH)" ]; then \
		curl -fL --retry 3 -o "$(MODEL_PATH)" "$(MODEL_URL)"; \
	fi
	@echo "$(MODEL_SHA256)  $(MODEL_PATH)" | shasum -a 256 -c
