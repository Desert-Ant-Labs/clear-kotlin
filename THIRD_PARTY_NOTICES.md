# Third-party notices: clear-kotlin

The `clear` models bundled in this library (`clear-studio`, `clear-natural`, in `library/src/main/assets/`) are a fine-tune of an upstream architecture. The upstream license applies to that base; nothing in the Desert Ant Labs Source-Available License overrides it.

For training-time dependencies and data sources see [`THIRD_PARTY_NOTICES.md` in clear-training](https://github.com/Desert-Ant-Labs/clear-training/blob/main/hf/THIRD_PARTY_NOTICES.md). This file covers only what's shipped at inference time.

## Runtime model components

### DeepFilterNet 3 (Hendrik Schröter)
- **Source:** [github.com/Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) (DFN3-half configuration).
- **License:** MIT License.
- **Use:** Base architecture for both `clear-studio` and `clear-natural`. Fine-tuned on the Desert Ant Labs speech corpus and exported to ONNX. The bundled `.onnx` files are bit-identical to the trained graph within fp16 precision.

## Runtime library dependencies

Pulled in transitively by the published `ai.desertant:clear` / `ai.desertant:clear-dsp` artifacts and present at runtime in consuming apps.

### ONNX Runtime Mobile (Microsoft)
- **Artifact:** `com.microsoft.onnxruntime:onnxruntime-android`.
- **Source:** [github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime).
- **License:** MIT License.
- **Use:** Executes the ONNX model graph (CPU / XNNPACK).

### JTransforms (Piotr Wendykier)
- **Artifact:** `com.github.wendykierp:JTransforms:3.1`.
- **Source:** [github.com/wendykierp/JTransforms](https://github.com/wendykierp/JTransforms).
- **License:** BSD 2-Clause License.
- **Use:** Mixed-radix FFT backing the 960-point STFT in the `:dsp` module.

### Kotlin Coroutines (JetBrains)
- **Artifact:** `org.jetbrains.kotlinx:kotlinx-coroutines-android`.
- **Source:** [github.com/Kotlin/kotlinx.coroutines](https://github.com/Kotlin/kotlinx.coroutines).
- **License:** Apache License 2.0.
- **Use:** Async execution of the enhancement pipeline.

The STFT/ISTFT geometry, ERB filterbank, feature extraction, WAV codec, and R128 loudness mastering are original Desert Ant Labs implementations with no third-party code.

## License-notice retention

### DeepFilterNet 3 (MIT)

```
MIT License

Copyright (c) 2022 Hendrik Schröter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The DeepFilterNet 3 notice also ships inside `huggingface.co/desert-ant-labs/clear` alongside the model weights.

### JTransforms (BSD 2-Clause)

```
Copyright (c) 2007 onward, Piotr Wendykier
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

ONNX Runtime Mobile (MIT) and Kotlin Coroutines (Apache 2.0) ship their own license texts within their respective artifacts.
