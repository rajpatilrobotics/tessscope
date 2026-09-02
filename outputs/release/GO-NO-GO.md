# TessScope local release-candidate audit

**Decision: GO for a local release candidate.**

This decision applies only to the repository package and judge artifacts. It does not
authorize a push, public repository, deployment, video upload, social post, hackathon
submission, or access to the locked BBBC006 test.

## Candidate under test

- Package milestone: local commit `b655405` (`feat: assemble Track 05 submission package`)
- Audit date: 2026-09-02
- Track: Tesseract Hackathon 2026, Track 05 — Differentiable graphics & rendering
- Public evidence scope: cached validation replay; `test_accessed=false`

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| Dependency lock | PASS | `uv lock --check`; 142 packages resolved |
| Code quality | PASS | repository-wide `uv run ruff check .` |
| Test suite | PASS | 209 tests; four documented upstream PyTorch/InstanSeg warnings |
| Judge bundle | PASS | `python3 scripts/serve_demo.py --check`; complete cached replay, no test access |
| Clean export | PASS | archive of `b655405` served over loopback; root, JavaScript, site manifest, and figure manifest returned successfully |
| Export isolation | PASS | no `.git`, `.venv`, external dataset tree, external model cache, download, Docker, retraining, or live inference |
| Links and paths | PASS | release audit found no broken repository-relative links or private/absolute project paths |
| License and attribution | PASS | exact Apache-2.0 license plus NOTICE, third-party notices, and citation metadata |
| Secrets and history | PASS | current tree and all 611 reachable history blobs scanned; no blockers |
| Repository size | PASS | 454 files and 63,046,046 bytes at the package milestone; largest current/history file is the 13,035,355-byte validation replay |
| Technical brief | PASS | four-page letter PDF; every page visually inspected; SHA-256 `9e232580ebb005e987ecb14a72d6dc6c65aa5d68f83d36ca7434468a074bd9d4` |
| Demo video | PASS | 210 s, 1920×1080, 30 fps, H.264/yuv420p with AAC stereo; eight timepoints visually inspected; SHA-256 `767f5d03ae859d68b4d06b94a1398beda1a38026e1549bdb7b04606254a275a6` |
| Determinism | PASS | the source-date-stabilized PDF and a second full video build reproduced byte-for-byte |
| Scientific boundary | PASS | validation-only claims remain qualified; exact-versus-piecewise is non-significant; test sealed; no physical-scope claim |

## Known non-blocking limitations

- The video is intentionally caption-led with a silent compatibility audio track; no
  unapproved synthetic voice or music is included.
- The instant judge path is a traced cached replay, not live model inference.
- The full scientific reproduction needs official external assets, Python 3.12, three
  local Tesseract services, and more compute than the instant demo.
- Exact closed-loop optimization significantly beats its matched stopped-gradient
  control, but does not significantly beat the strongest piecewise baseline.

## External actions still requiring separate approval

1. Create or choose a public GitHub repository and push the local release candidate.
2. Upload the video and any required submission media.
3. Submit the project to the hackathon and publish any associated post.

None of those external actions was performed during this audit.
