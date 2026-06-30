# Keep the public API so R8/ProGuard in consuming apps doesn't strip
# entry points the library author guarantees.
-keep public class ai.desertant.clear.Clear { *; }
-keep public class ai.desertant.clear.Clear$* { *; }

# ONNX Runtime uses reflection to load native bindings.
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
