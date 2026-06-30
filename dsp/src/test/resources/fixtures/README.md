# Numerical-parity fixtures

Per-stage outputs dumped from the Swift reference for the Kotlin parity
tests.

## Layout

```
fixtures/
  S1_stft_forward/    case_NNN_<name>.clear-fixture
  S2_erb_features/
  S3_spec_features/
  S5_stft_inverse/
```

Each `.clear-fixture` carries one input + the Swift-produced output(s)
for one stage and one synthetic test case (sine 440 Hz, sine 1 kHz,
sine 8 kHz, log sweep 100 Hz – 8 kHz, silence, white noise).

## Why these are committed

These fixtures pin the Swift ↔ Kotlin contract. They're not test inputs
in the usual sense — they encode the expected behavior the Kotlin
port must match. Treat them like golden files.

If a Kotlin port change makes the tests fail, the fix is in Kotlin, not
in the fixtures. Only regenerate when the Swift side changed
intentionally.
