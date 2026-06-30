package ai.desertant.clear.parity

import ai.desertant.clear.internal.Constants
import ai.desertant.clear.internal.dsp.FeatureExtractor
import org.junit.Test

/**
 * Stage S2 — ERB log-energy features with stateful EMA normalization.
 *
 * Exercises the EMA ramp initialization (the documented zero-init
 * footgun). The first ~200 frames are where mismatches show up if
 * the init is wrong; the synthetic cases are 25 frames each, which
 * is squarely inside that window.
 */
class Stage2ErbFeaturesTest {

    @Test
    fun `S2 ERB features match Swift on all synthetic cases`() {
        val fixtures = Fixture.loadStage("S2_erb_features")
        require(fixtures.isNotEmpty()) { "no S2 fixtures on classpath" }

        var caseCount = 0
        for (fx in fixtures) {
            val inReal = fx.tensor("input_real")
            val inImag = fx.tensor("input_imag")
            val expected = fx.tensor("feat_erb")
            val nFrames = inReal.shape[0]
            require(inReal.shape[1] == Constants.N_FREQ)

            val out = FeatureExtractor.compute(inReal.data, inImag.data, nFrames)
            assertArrayClose(expected.data, out.featErb,
                absTol = fx.toleranceAbs, relTol = fx.toleranceRel,
                label = "${fx.caseName} featErb")
            caseCount++
        }
        println("S2 ERB features: $caseCount cases passed")
    }
}
