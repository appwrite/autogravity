# Image fixtures

These files are copied unchanged from `golang.org/x/image` v0.45.0,
`testdata/`. The upstream BSD license is included in `LICENSE`.

| Local name | Upstream name | Purpose |
| --- | --- | --- |
| rose.png | yellow_rose.png | Landscape flower photograph, PNG with transparency |
| rose-lossless.webp | yellow_rose.lossless.webp | Lossless WebP of the flower |
| rose-lossy.webp | yellow_rose.lossy.webp | Lossy WebP of the flower |
| rose-alpha.webp | yellow_rose.lossy-with-alpha.webp | Lossy WebP with alpha |
| portrait.jpg | go-turns-two-280x360.jpeg | Portrait photograph of a Go plush toy |

Source: https://go.googlesource.com/image/+/refs/tags/v0.45.0/testdata/

Tests read these fixtures locally and require no downloads. EXIF tests inject
orientation metadata into the JPEG without re-encoding its pixels. Model tests
use broad subject regions and transformation consistency, rather than exact
floating-point snapshots, to tolerate runtime and architecture differences.
