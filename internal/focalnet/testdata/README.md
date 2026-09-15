# testdata

`dummy.onnx` is a format-v2 placeholder: inputs `image` `[1,3,256,256]`,
`boxes` `[1,128,4]`, `content` `[1,4]`; outputs `importance` `[1,1,64,64]` and
`crop_scores` `[1,128]`. The map is a fixed centered Gaussian. Crop scores
prefer boxes whose center is closest to the image midpoint and ignore image
pixels. It exists so CI can exercise the FocalNet session path without the
evaluated 19 MiB checkpoint. Do not use it for quality judgments.

`letterboxes.json` is generated from FocalNet's Python `Letterbox.fit` so the
Go port stays aligned with `appwrite/focalnet`.

`ranking.json` is generated from FocalNet's Python `generate_candidates`,
`letterbox_content`, and `score_crop` helpers.
