# Library assets

The Android library bundles its ONNX models here. The Gradle build picks
them up automatically (anything in `assets/` ships with the AAR).

## v0.2

| File | Size | Source of truth |
|---|---:|---|
| `clear-studio.onnx` | ~24 MB fp16 | `clear-training/models/weights/clear-studio.onnx` |
| `clear-natural.onnx` | ~24 MB fp16 | `clear-training/models/weights/clear-natural.onnx` |

Both distilled from their respective teachers via layer-wise TCN
supervision (see clear-training's `models/distill/`). I/O contract:
`spec / feat_erb / feat_spec → spec_enhanced`, same as v0.1.

For dev against an unshipped checkpoint, set `CLEAR_MODEL_LOCAL_PATH`
(mirrors the iOS `CLEAR_MODEL_LOCAL_DIR` pattern).
