# TessScope v2.5 audit

## Claim boundary

V2.5 demonstrates that the registered exact constrained implementation, B11 derivative,
B11 primary matrix, B7 continuity matrix, and both conditional SLSQP fallbacks ran to
their frozen limits. It does not demonstrate a promoted B11 pupil or a positive test
result.

## Ordering audit

1. Contract and source hashes were frozen before a new pupil was generated.
2. The full B11 derivative passed before constrained optimization.
3. Every B11 and B7 primary start/bound combination completed its 18-step budget.
4. Zero primary endpoint was eligible, so the registered fallback activated for both
   bases and used the frozen two-batch constraint anchors.
5. Zero B11 endpoint passed the training and 12-well soft gates.
6. Piecewise, stopped-stage, hard, derivative-free, gain, and test work did not activate.
7. `test_accessed` is false in the derivative, B11, B7, and final decision artifacts.

## Corrected smoke-run audit

Two invalid partial smoke attempts were stopped before scientific use. The first used a
per-minibatch constraint where the contract required the four-batch aggregate. The
second compared a two-batch fallback value to a four-batch anchor. Their diagnostic
records are preserved under `artifacts/runs/v2_5/diagnostics/`; neither contributed a
candidate or changed a gate. The valid runs use the exact aggregate primary objective
and matched two-batch fallback anchor.

## Final integrity source

`configs/v2_5/pretest-block.json` contains the SHA-256 hashes of the contract, source
manifest, derivative evidence, and both final optimization matrices. Automated tests
recompute those hashes and assert zero promotions and a sealed test.
