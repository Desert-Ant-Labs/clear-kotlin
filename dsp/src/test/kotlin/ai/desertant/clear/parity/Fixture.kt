package ai.desertant.clear.parity

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.json.JSONArray
import org.json.JSONObject

/**
 * In-memory representation of a `.clear-fixture` binary file.
 *
 * File format:
 *   [ 8 bytes ]  magic              = "CLRFXTR\0"
 *   [ 4 bytes ]  metadata_len (u32 LE)
 *   [ M bytes ]  metadata           UTF-8 JSON
 *   [ remaining] tensor data        packed back-to-back, raw LE float32
 */
data class Fixture(
    val stage: String,
    val caseName: String,
    val swiftVersion: String,
    val modelVariant: String,
    val createdUtc: String,
    val toleranceAbs: Float,
    val toleranceRel: Float,
    val tensors: Map<String, Tensor>,
) {
    data class Tensor(val shape: IntArray, val data: FloatArray) {
        override fun equals(other: Any?): Boolean =
            other is Tensor && shape.contentEquals(other.shape) && data.contentEquals(other.data)
        override fun hashCode(): Int = 31 * shape.contentHashCode() + data.contentHashCode()
    }

    fun tensor(name: String): Tensor =
        tensors[name] ?: error("fixture $stage/$caseName missing tensor '$name'; have ${tensors.keys}")

    companion object {
        // Magic bytes: "CLRFXTR" + 0x00. Compared byte-wise.
        private val MAGIC = byteArrayOf(0x43, 0x4C, 0x52, 0x46, 0x58, 0x54, 0x52, 0x00)

        fun load(file: File): Fixture {
            val bytes = file.readBytes()
            require(bytes.size >= 12) { "fixture too small: ${file.path}" }
            for (i in 0..7) {
                require(bytes[i] == MAGIC[i]) { "bad magic byte $i in ${file.path}" }
            }
            val metaLen = ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int
            require(metaLen in 1..(bytes.size - 12)) {
                "bad metadata length $metaLen in ${file.path}"
            }
            val metaJson = String(bytes, 12, metaLen, Charsets.UTF_8)
            val meta = JSONObject(metaJson)
            val tensorSpecs: JSONArray = meta.getJSONArray("tensors")
            val payload = ByteBuffer.wrap(bytes, 12 + metaLen, bytes.size - 12 - metaLen)
                .order(ByteOrder.LITTLE_ENDIAN)

            val tensors = LinkedHashMap<String, Tensor>()
            for (i in 0 until tensorSpecs.length()) {
                val spec = tensorSpecs.getJSONObject(i)
                val name = spec.getString("name")
                val dtype = spec.getString("dtype")
                require(dtype == "f32") { "unsupported dtype $dtype" }
                val shapeJson = spec.getJSONArray("shape")
                val shape = IntArray(shapeJson.length()) { shapeJson.getInt(it) }
                val count = shape.fold(1) { acc, dim -> acc * dim }
                val data = FloatArray(count)
                for (j in 0 until count) data[j] = payload.float
                tensors[name] = Tensor(shape, data)
            }

            return Fixture(
                stage = meta.getString("stage"),
                caseName = meta.getString("case"),
                swiftVersion = meta.getString("swift_version"),
                modelVariant = meta.getString("model_variant"),
                createdUtc = meta.getString("created_utc"),
                toleranceAbs = meta.getDouble("tolerance_abs").toFloat(),
                toleranceRel = meta.getDouble("tolerance_rel").toFloat(),
                tensors = tensors,
            )
        }

        /**
         * Load every fixture in a stage directory under
         * `src/test/resources/fixtures/<stage>/`.
         */
        fun loadStage(stage: String): List<Fixture> {
            val dir = stageDir(stage)
            require(dir.exists() && dir.isDirectory) {
                "fixture stage dir missing: ${dir.path}\n" +
                "Committed parity fixtures should live under dsp/src/test/resources/fixtures/."
            }
            return dir.listFiles { f -> f.name.endsWith(".clear-fixture") }
                ?.sortedBy { it.name }
                ?.map(::load)
                ?: emptyList()
        }

        fun stageDir(stage: String): File {
            val resource = Fixture::class.java.classLoader
                ?.getResource("fixtures/$stage")
                ?: error(
                    "fixtures/$stage not on the test classpath. " +
                    "Committed parity fixtures should live under dsp/src/test/resources/fixtures/."
                )
            return File(resource.toURI())
        }
    }
}
