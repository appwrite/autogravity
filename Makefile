FP32_MODEL_PATH := models/u2net.onnx
FP32_MODEL_URL := https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
FP32_MODEL_SHA256 := 8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491
FACE_MODEL_PATH := models/face_detection_yunet_2023mar.onnx
FACE_MODEL_SHA256 := 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
FOCALNET_MODEL_PATH := models/focalnet-human.onnx
FOCALNET_MODEL_URL := https://github.com/appwrite/focalnet/releases/download/2026-09-14-rc1/focalnet-human.onnx
FOCALNET_MODEL_TAG := 2026-09-14-rc1
FOCALNET_MODEL_SHA256 := 59164c601c98cea3f62b25166710831dac63e1a872fc64767c65316ad5385439
VERSION ?= dev
COMMIT ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)
LDFLAGS := -X main.version=$(VERSION) -X main.commit=$(COMMIT)

.PHONY: build run test test-integration test-integration-fp32 evaluate model model-fp32 model-int8 model-face model-focalnet

build:
	go build -ldflags="$(LDFLAGS)" -o autogravity ./cmd/autogravity

run: model
	go run ./cmd/autogravity

test:
	go test ./...

test-integration: model
	go test -race -tags=integration ./...

test-integration-fp32: model-fp32 model-face
	MODEL_PATH= MODEL_PRECISION=fp32 go test -race -tags=integration ./...

evaluate: model
	go test -tags=integration,evaluation -run TestEvaluateDifficultScenes -v ./cmd/autogravity

model: model-fp32 model-int8 model-face

model-int8:
	@echo "b340186f56660b6665e494aab912e5f8e9adbc2317181c77fd01aa226f06553b  models/u2net-int8.onnx" | shasum -a 256 -c

model-face:
	@echo "$(FACE_MODEL_SHA256)  $(FACE_MODEL_PATH)" | shasum -a 256 -c

model-fp32:
	@if [ ! -f "$(FP32_MODEL_PATH)" ]; then \
		curl -fL --retry 3 -o "$(FP32_MODEL_PATH)" "$(FP32_MODEL_URL)"; \
	fi
	@echo "$(FP32_MODEL_SHA256)  $(FP32_MODEL_PATH)" | shasum -a 256 -c

model-focalnet:
	@if [ ! -f "$(FOCALNET_MODEL_PATH)" ]; then \
		if command -v gh >/dev/null 2>&1 && gh release download "$(FOCALNET_MODEL_TAG)" --repo appwrite/focalnet --pattern focalnet-human.onnx --dir models --clobber; then \
			true; \
		else \
			curl -fL --retry 3 -o "$(FOCALNET_MODEL_PATH)" "$(FOCALNET_MODEL_URL)"; \
		fi; \
	fi
	@echo "$(FOCALNET_MODEL_SHA256)  $(FOCALNET_MODEL_PATH)" | shasum -a 256 -c
