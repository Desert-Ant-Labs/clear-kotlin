package ai.desertant.clear.parity

import ai.desertant.clear.internal.dsp.Stft
import org.junit.Test

/**
 * Stage S5 — Inverse STFT. Reconstructs time-domain samples from the
 * (real, imag) complex spectrum that Stage S1 produced. End-to-end
 * round-trip approximation: S5(S1(x)) ≈ x within abs 1e-5.
 *
 * Also exercises the spectrum-mirror step (DC-to-Nyquist → full N bins)
 * which is the single most fiddly index manipulation in the iSTFT.
 */
class Stage5StftInverseTest {

    @Test
    fun `S5 STFT inverse matches Swift on all synthetic cases`() {
        val fixtures = Fixture.loadStage("S5_stft_inverse")
        require(fixtures.isNotEmpty()) { "no S5 fixtures on classpath" }

        val stft = Stft()
        var caseCount = 0
        for (fx in fixtures) {
            val inReal = fx.tensor("input_real")
            val inImag = fx.tensor("input_imag")
            val expected = fx.tensor("output")
            val nFrames = inReal.shape[0]

            val out = stft.inverse(inReal.data, inImag.data, nFrames)
            assert(out.size == expected.data.size) {
                "${fx.caseName}: output size ${out.size} ≠ ${expected.data.size}"
            }
            assertArrayClose(expected.data, out,
                absTol = fx.toleranceAbs, relTol = fx.toleranceRel,
                label = "${fx.caseName} reconstructed")
            caseCount++
        }
        println("S5 STFT inverse: $caseCount cases passed")
    }
}
