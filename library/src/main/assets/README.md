# Library assets

The Android library bundles both model variants here as raw assets:

| File | Size | Variant |
|---|---|---|
| `clear-studio.onnx` | ~4.5 MB | default; quiet, studio-like cleanup |
| `clear-natural.onnx` | ~4.5 MB | preserves room tone, breath, lip texture |

Both are 6-bit palettized (fp16-stored) ONNX exports of the published
`desert-ant-labs/clear` weights. The Gradle build picks them up
automatically (anything in `assets/` ships with the AAR), so the library
works offline with no first-launch download.

`Clear.ModelVariant` selects which one loads (`ClearStudio` by default).

For development against an unshipped checkpoint, set
`CLEAR_MODEL_LOCAL_PATH` (a future env var to be wired in
`Clear.invoke()`) so the library loads the local file instead of the
bundled asset — mirrors the iOS `CLEAR_MODEL_LOCAL_DIR` pattern.
