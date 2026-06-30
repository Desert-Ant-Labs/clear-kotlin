package ai.desertant.clear.internal

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import ai.desertant.clear.internal.dsp.ModelBridge
import java.io.Closeable
import java.nio.FloatBuffer

/**
 * ONNX Runtime Mobile bridge. Runs the bundled Clear model on the
 * CPU (XNNPACK) execution provider.
 */
internal class OnnxModelBridge private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
) : ModelBridge, Closeable {

    private val tShape: LongArray = longArrayOf(1, 1, Constants.CHUNK_LEN.toLong(), Constants.N_FREQ.toLong(), 2)
    private val erbShape: LongArray = longArrayOf(1, 1, Constants.CHUNK_LEN.toLong(), Constants.N_ERB.toLong())
    private val featSpecShape: LongArray = longArrayOf(1, 1, Constants.CHUNK_LEN.toLong(), Constants.N_DF.toLong(), 2)

    private val outElementCount: Int = Constants.CHUNK_LEN * Constants.N_FREQ * 2

    override fun runChunk(spec: FloatArray, featErb: FloatArray, featSpec: FloatArray): FloatArray {
        require(spec.size == Constants.CHUNK_LEN * Constants.N_FREQ * 2)
        require(featErb.size == Constants.CHUNK_LEN * Constants.N_ERB)
        require(featSpec.size == Constants.CHUNK_LEN * Constants.N_DF * 2)

        OnnxTensor.createTensor(env, FloatBuffer.wrap(spec), tShape).use { specT ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(featErb), erbShape).use { erbT ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(featSpec), featSpecShape).use { fsT ->
                    val inputs = mapOf("spec" to specT, "feat_erb" to erbT, "feat_spec" to fsT)
                    session.run(inputs).use { result ->
                        val out = result[0] as OnnxTensor
                        val buf = out.floatBuffer
                        val arr = FloatArray(outElementCount)
                        buf.get(arr)
                        return arr
                    }
                }
            }
        }
    }

    override fun close() {
        session.close()
        // OrtEnvironment is a process-wide singleton — do not close.
    }

    public companion object {
        /** Load a bundled ONNX model (default `clear-studio.onnx`) from the app's assets. */
        public fun fromAssets(context: Context, assetName: String = "clear-studio.onnx"): OnnxModelBridge {
            val bytes = context.assets.open(assetName).use { it.readBytes() }
            return fromBytes(bytes)
        }

        public fun fromBytes(bytes: ByteArray): OnnxModelBridge {
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                // 4 ≈ performance-core count on most big.LITTLE Android SoCs;
                // oversubscribing onto little cores halves throughput.
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
                // ALL_OPT measured ~18% faster than BASIC_OPT on-device.
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }
            val session = env.createSession(bytes, opts)
            return OnnxModelBridge(env, session)
        }
    }
}
