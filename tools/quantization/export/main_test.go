package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestValidateInputs(t *testing.T) {
	dir := t.TempDir()
	a, b := filepath.Join(dir, "a"), filepath.Join(dir, "b")
	for path, content := range map[string]string{a: "first", b: "second"} {
		if err := os.WriteFile(path, []byte(content), 0600); err != nil {
			t.Fatal(err)
		}
	}
	valid := []input{{"a.jpg", a, "calibration", "a"}, {"b.jpg", b, "evaluation", "b"}}
	if err := validateInputs(valid); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"duplicate bytes", "duplicate name", "cross split group", "missing group", "invalid split", "unsafe name", "missing split"} {
		t.Run(name, func(t *testing.T) {
			items := append([]input(nil), valid...)
			switch name {
			case "duplicate bytes":
				items[1].Path = a
			case "duplicate name":
				items[1].Name = items[0].Name
			case "cross split group":
				items[1].Group = items[0].Group
			case "missing group":
				items[1].Group = ""
			case "invalid split":
				items[1].Split = "test"
			case "unsafe name":
				items[1].Name = "../b.jpg"
			case "missing split":
				items = items[:1]
			}
			if err := validateInputs(items); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestLoadInputsRelativePaths(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "dataset.json")
	if err := os.WriteFile(manifest, []byte(`[{"name":"a.jpg","path":"photos/a.jpg","split":"calibration","group":"a"}]`), 0600); err != nil {
		t.Fatal(err)
	}
	items, err := loadInputs(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if items[0].Path != filepath.Join(dir, "photos/a.jpg") {
		t.Fatalf("unexpected path: %s", items[0].Path)
	}
}
