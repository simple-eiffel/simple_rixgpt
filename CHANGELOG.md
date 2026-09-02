# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-02

First public release. RixGPT itself was written and shipped as a Windows
installer on **2026-07-14**; it had never been under version control. This
release brings the source into git and onto the simple_* fleet pattern. No
behavior was changed.

### Added
- `README.md`, `CHANGELOG.md`, `LICENSE`, `package.json`, `.gitignore`, and a
  documentation site under `docs/` published to GitHub Pages.
- `simple_rixgpt_tests` target and `testing/test_app.e`. It is a compile proof
  rather than behavioral coverage, and it says so in its own class note:
  `RIXGPT_MODEL`'s only creation procedure takes a path to a weights file, so
  nothing in the class can be exercised without weights this repository does not
  carry. What the target does prove is that both classes type-check and
  finalize.
- `<description>` on the ECF system.
- `.eiffel-workflow/evidence/fleet-onboarding.txt` — build command, clean-build
  output, the stale-executable timestamp check, and the warning grep.

### Verified, not changed
- Clean release build of `simple_rixgpt_app` after `rm -rf EIFGENs/simple_rixgpt_app`:
  exit 0, zero errors, zero compiler warnings, executable timestamp newer than
  the build start.
- The header the README documents was read back out of the shipped `model.bin`
  rather than assumed: format version 2, `dtype` 1 (fp16), 6 layers, 6 heads,
  `n_embd` 384, vocabulary 316, context 256. The file's size accounts for exactly
  10,867,200 fp16 weights over a 1,300-byte header, with no remainder.

### Not in the repository
The trained weights (`model.bin`), the runnable `bin/` folder, the installer
staging folder (`installer/src/`) and the compiled `RixGPT-Setup.exe` are all
excluded by `.gitignore`. They are build payloads produced by Larry Rix's
private training pipeline, not source. `installer/RixGPT.iss` is source and is
tracked.

### Known limitations
- **No model, no run.** Without `model.bin` the program starts, reports the
  missing path and exits. Nothing in this repository can generate text on its
  own.
- **The model imitates a voice; it does not know anything.** A 10.9 M character
  model produces fluent invention, including invented citations. The startup
  banner and `--help` both say so.
- **Single-threaded and unvectorized.** Every matrix-vector product is a scalar
  Eiffel loop over a `REAL_64` array. A prompt takes a few seconds on the CPU.
  That is the price of the design goal, which was to have no dependency to
  install rather than to be fast.
- **Context is the model's, not the program's.** Generation stops at
  `block_size` positions — 256 characters for the shipped weights — counting the
  prompt. There is no sliding window and no cache eviction.
- **fp16 infinities and NaNs decode to zero.** `half_to_double` maps the
  all-ones exponent to `0.0`. Trained weights do not contain those patterns, so
  this has never mattered, but a corrupt file would be read as zeros rather than
  rejected.
- **The loader trusts the file.** There is no checksum and no bounds check
  against the declared shape; a truncated `model.bin` fails as an array
  index-out-of-bounds rather than as a diagnosis.

[0.1.0]: https://github.com/simple-eiffel/simple_rixgpt/releases/tag/v0.1.0
