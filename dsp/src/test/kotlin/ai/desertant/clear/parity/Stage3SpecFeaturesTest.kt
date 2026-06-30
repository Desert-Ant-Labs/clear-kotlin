package ai.desertant.clear.parity

import ai.desertant.clear.internal.Constants
import ai.desertant.clear.internal.dsp.FeatureExtractor
import org.junit.Test

/**
 * Stage S3 — Deep-filter pathway features. Per-band √RMS EMA over the
 * first 96 spectral bins.
 *
 * Exercises the second documented ramp init (`s = linspace(1e-3, 1e-4)`).
 * Zero-init here makes the first 1/sqrt(s) divide explode.
 */
class Stage3SpecFeaturesTest {

    @Test
    fun `S3 spec features match Swift on all synthetic cases`() {
        val fixtures = Fixture.loadStage("S3_spec_features")
        require(fixtures.isNotEmpty()) { "no S3 fixtures on classpath" }

        var caseCount = 0
        for (fx in fixtures) {
            val inReal = fx.tensor("input_real")
            val inImag = fx.tensor("input_imag")
            val expectedReal = fx.tensor("feat_spec_real")
            val expectedImag = fx.tensor("feat_spec_imag")
            val nFrames = inReal.shape[0]
            require(inReal.shape[1] == Constants.N_FREQ)
            require(expectedReal.shape[1] == Constants.N_DF)

            val out = FeatureExtractor.compute(inReal.data, inImag.data, nFrames)
            assertArrayClose(expectedReal.data, out.featSpecReal,
                absTol = fx.toleranceAbs, relTol = fx.toleranceRel,
                label = "${fx.caseName} featSpecReal")
            assertArrayClose(expectedImag.data, out.featSpecImag,
                absTol = fx.toleranceAbs, relTol = fx.toleranceRel,
                label = "${fx.caseName} featSpecImag")
            caseCount++
        }
        println("S3 spec features: $caseCount cases passed")
    }
}
