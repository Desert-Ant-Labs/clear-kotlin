package ai.desertant.clear.parity

import ai.desertant.clear.internal.dsp.Stft
import org.junit.Test

/**
 * Stage S1 — Forward STFT. Compares Kotlin `Stft.forward(samples)` to
 * the Swift reference output for every committed synthetic case.
 *
 * Tolerance: 1e-5 abs, 1e-6 rel — JTransforms and vDSP are both
 * float32 implementations of the same Cooley-Tukey algorithm, so
 * agreement should be tight modulo accumulation order.
 */
class Stage1StftForwardTest {

    @Test
    fun `S1 STFT forward matches Swift on all synthetic cases`() {
        val fixtures = Fixture.loadStage("S1_stft_forward")
        require(fixtures.isNotEmpty()) { "no S1 fixtures on classpath" }

        val stft = Stft()
        var caseCount = 0
        for (fx in fixtures) {
            val input = fx.tensor("input").data
            val refReal = fx.tensor("output_real")
            val refImag = fx.tensor("output_imag")
            val expectedShape = refReal.shape

            val out = stft.forward(input)
            assert(out.nFrames == expectedShape[0]) {
                "${fx.caseName}: nFrames ${out.nFrames} ≠ ${expectedShape[0]}"
            }

            assertArrayClose(refReal.data, out.real,
                absTol = fx.toleranceAbs, relTol = fx.toleranceRel,
                label = "${fx.caseName} real")
            assertArrayClose(refImag.data, out.imag,
                absTol = fx.toleranceAbs, relTol = fx.toleranceRel,
                label = "${fx.caseName} imag")
            caseCount++
        }
        println("S1 STFT forward: $caseCount cases passed")
    }
}
