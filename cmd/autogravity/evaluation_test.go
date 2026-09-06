//go:build integration && evaluation

package main

import "testing"

// These manually annotated natural scenes currently expose model failures.
// Keep them executable without accepting incorrect outputs as golden values.
// Run with make evaluate; this is intentionally separate from the CI gate.
func TestEvaluateDifficultScenes(t *testing.T) {
	runSubjectCases(t, []subjectCase{
		{"person-room.jpg", 0.40, 0.58, 0.55, 0.90},
		{"pedestrian-dog.jpg", 0.55, 0.94, 0.57, 0.80},
		{"bird-branch.jpg", 0.27, 0.34, 0.32, 0.40},
	})
}
