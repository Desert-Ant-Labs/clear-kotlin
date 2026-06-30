package ai.desertant.clear.parity

import kotlin.math.abs
import kotlin.math.max
import org.junit.Assert.fail

/**
 * Compare a Kotlin-produced array to the Swift reference within a
 * declared tolerance. Fails with a useful diff (the first 5 offending
 * elements) when they disagree.
 *
 * Tolerance semantics: an element passes if `|a - b| <= absTol` OR
 * `|a - b| <= relTol * max(|a|, |b|)`. This matches the numpy
 * `assert_allclose` convention.
 */
fun assertArrayClose(
    expected: FloatArray,
    actual: FloatArray,
    absTol: Float,
    relTol: Float = 0f,
    label: String = "array",
) {
    if (expected.size != actual.size) {
        fail("$label: size mismatch — expected ${expected.size}, got ${actual.size}")
        return
    }
    val mismatches = mutableListOf<Triple<Int, Float, Float>>()  // (idx, expected, actual)
    var maxAbsErr = 0f
    var maxAbsErrAt = -1
    for (i in expected.indices) {
        val e = expected[i]
        val a = actual[i]
        val absErr = abs(e - a)
        if (absErr > maxAbsErr) { maxAbsErr = absErr; maxAbsErrAt = i }
        val tol = max(absTol, relTol * max(abs(e), abs(a)))
        if (absErr > tol && !(e.isNaN() && a.isNaN())) {
            if (mismatches.size < 5) mismatches.add(Triple(i, e, a))
        }
    }
    if (mismatches.isNotEmpty()) {
        val diffStr = mismatches.joinToString("\n") { (i, e, a) ->
            "  [%d] expected=%.6g actual=%.6g  |Δ|=%.3g".format(i, e, a, abs(e - a))
        }
        fail("""
            |$label: ${mismatches.size}+ elements outside tolerance (abs=$absTol, rel=$relTol).
            |Max |Δ| = $maxAbsErr at index $maxAbsErrAt (of ${expected.size}).
            |First offenders:
            |$diffStr
            """.trimMargin())
    }
}
