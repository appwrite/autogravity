# testdata

`dummy.onnx` is a format-v1 placeholder: input `image` `[1,3,256,256]`, output
`importance` `[1,1,64,64]`. The map is a fixed centered Gaussian and ignores
image content. It exists so CI can exercise the FocalNet session path without
the evaluated 19 MiB checkpoint. Do not use it for quality judgments.

`letterboxes.json` is generated from FocalNet's Python `Letterbox.fit` so the
Go port stays aligned with `appwrite/focalnet`.
