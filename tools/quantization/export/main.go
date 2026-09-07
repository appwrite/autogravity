// Export uses the production decoder and normalization for quantization inputs.
// Run from the repository root: go run ./tools/quantization/export -out DIR
package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"autogravity/internal/imageutil"
)

type fixture struct {
	Name   string `json:"name"`
	Split  string `json:"split"`
	Tensor string `json:"tensor"`
	Region [4]int `json:"region"`
	SHA256 string `json:"sha256"`
}

// Group identifies a source photo, including its resized/re-encoded variants.
type input struct {
	Name  string `json:"name"`
	Path  string `json:"path"`
	Split string `json:"split"`
	Group string `json:"group"`
}

func loadInputs(manifest string) ([]input, error) {
	if manifest == "" {
		var items []input
		for _, pair := range [][2]string{
			{"rose.png", "calibration"}, {"portrait.jpg", "calibration"},
			{"puppies.jpg", "calibration"}, {"person-room.jpg", "calibration"},
			{"dog-portrait.jpg", "evaluation"}, {"panda-bamboo.jpg", "evaluation"},
			{"pedestrian-dog.jpg", "evaluation"}, {"bird-branch.jpg", "evaluation"},
		} {
			items = append(items, input{pair[0], filepath.Join("internal/testimages/testdata", pair[0]), pair[1], pair[0]})
		}
		return items, nil
	}
	data, err := os.ReadFile(manifest)
	if err != nil {
		return nil, err
	}
	var items []input
	if err := json.Unmarshal(data, &items); err != nil {
		return nil, err
	}
	for i := range items {
		if !filepath.IsAbs(items[i].Path) {
			items[i].Path = filepath.Join(filepath.Dir(manifest), items[i].Path)
		}
	}
	return items, nil
}

func validateInputs(items []input) error {
	names, hashes, groups := map[string]bool{}, map[[32]byte]string{}, map[string]string{}
	counts := map[string]int{}
	for _, item := range items {
		if item.Name == "" || filepath.Base(item.Name) != item.Name || item.Name == "." || item.Name == ".." {
			return fmt.Errorf("invalid fixture name %q", item.Name)
		}
		if names[item.Name] {
			return fmt.Errorf("duplicate fixture name %q", item.Name)
		}
		names[item.Name] = true
		if item.Split != "calibration" && item.Split != "evaluation" {
			return fmt.Errorf("invalid split for %q", item.Name)
		}
		if item.Group == "" {
			return fmt.Errorf("source-photo group is required for %q", item.Name)
		}
		if split, ok := groups[item.Group]; ok && split != item.Split {
			return fmt.Errorf("source-photo group %q crosses splits", item.Group)
		}
		groups[item.Group] = item.Split
		data, err := os.ReadFile(item.Path)
		if err != nil {
			return err
		}
		hash := sha256.Sum256(data)
		if other, ok := hashes[hash]; ok {
			return fmt.Errorf("duplicate image bytes: %q and %q", other, item.Name)
		}
		hashes[hash] = item.Name
		counts[item.Split]++
	}
	if counts["calibration"] == 0 || counts["evaluation"] == 0 {
		return fmt.Errorf("both calibration and evaluation images are required")
	}
	return nil
}

func main() {
	out := flag.String("out", "", "output directory for tensors and manifest")
	manifest := flag.String("manifest", "", "external dataset JSON; image paths relative to this file")
	flag.Parse()
	if *out == "" {
		panic("-out is required")
	}
	items, err := loadInputs(*manifest)
	if err != nil {
		panic(err)
	}
	if err := validateInputs(items); err != nil {
		panic(err)
	}
	if err := os.MkdirAll(*out, 0755); err != nil {
		panic(err)
	}
	var fixtures []fixture
	for _, item := range items {
		data, err := os.ReadFile(item.Path)
		if err != nil {
			panic(err)
		}
		img, err := imageutil.Decode(data)
		if err != nil {
			panic(err)
		}
		tensor, region, err := imageutil.Prepare(img, 320, 320)
		if err != nil {
			panic(err)
		}
		name := item.Name + ".f32"
		file, err := os.Create(filepath.Join(*out, name))
		if err != nil {
			panic(err)
		}
		if err := binary.Write(file, binary.LittleEndian, tensor); err != nil {
			panic(err)
		}
		if err := file.Close(); err != nil {
			panic(err)
		}
		fixtures = append(fixtures, fixture{item.Name, item.Split, name, [4]int{region.Min.X, region.Min.Y, region.Max.X, region.Max.Y}, fmt.Sprintf("%x", sha256.Sum256(data))})
	}
	data, err := json.MarshalIndent(fixtures, "", "  ")
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(filepath.Join(*out, "manifest.json"), data, 0644); err != nil {
		panic(err)
	}
	fmt.Printf("Exported %d production-preprocessed tensors to %s\n", len(fixtures), *out)
}
