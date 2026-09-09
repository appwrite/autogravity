// Package ortenv manages the process-wide ONNX Runtime environment.
package ortenv

import (
	"errors"
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var environment struct {
	sync.Mutex
	references  int
	libraryPath string
}

// Acquire initializes ONNX Runtime on the first call and retains the shared
// environment for subsequent model sessions. Every successful call must be
// paired with Release.
func Acquire(libraryPath string) error {
	if libraryPath == "" {
		return errors.New("ONNX Runtime library path is required")
	}

	environment.Lock()
	defer environment.Unlock()
	if environment.references > 0 {
		if libraryPath != environment.libraryPath {
			return fmt.Errorf("ONNX Runtime already initialized from %q", environment.libraryPath)
		}
		environment.references++
		return nil
	}
	if ort.IsInitialized() {
		return errors.New("ONNX Runtime was initialized outside the shared environment manager")
	}

	ort.SetSharedLibraryPath(libraryPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return err
	}
	environment.references = 1
	environment.libraryPath = libraryPath
	return nil
}

// Release drops a reference and destroys ONNX Runtime after the final model
// session has been closed.
func Release() error {
	environment.Lock()
	defer environment.Unlock()
	if environment.references == 0 {
		return errors.New("ONNX Runtime environment is not acquired")
	}

	environment.references--
	if environment.references > 0 {
		return nil
	}
	environment.libraryPath = ""
	return ort.DestroyEnvironment()
}
