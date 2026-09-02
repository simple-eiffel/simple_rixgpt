# simple_rixgpt

**[GitHub](https://github.com/simple-eiffel/simple_rixgpt)** •
[Documentation](https://simple-eiffel.github.io/simple_rixgpt/) •
[Issues](https://github.com/simple-eiffel/simple_rixgpt/issues)

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Eiffel 25.02](https://img.shields.io/badge/Eiffel-25.02-purple.svg)
![DBC: Contracts](https://img.shields.io/badge/DBC-Contracts-green.svg)

RixGPT is a character-level GPT whose entire forward pass is pure Eiffel:
`RIXGPT_MODEL` reads a self-describing `RIXG` weights file through `RAW_FILE` and
a `MANAGED_POINTER`, widens its fp16 tensors into a `REAL_64` buffer, and then
generates one character at a time through a KV-cached transformer — attention,
GELU, layernorm and top-k sampling all written out in Eiffel, with no Python, no
GPU and no C library beyond `KERNEL32.dll`. `RIXGPT_APP` wraps it in a console
REPL: type a few words, press Enter, and about 400 characters come back.

Part of the [Simple Eiffel](https://github.com/simple-eiffel) ecosystem.

## Status

> **0.1.0 — first public release.** The program was built and shipped as a
> Windows installer on **2026-07-14**; this release brings its source under
> version control and onto the simple_* fleet pattern. Clean release build,
> zero errors, zero compiler warnings.
>
> Be clear about what this is. It is **an inference engine**, not a training
> framework: there is no backward pass, no optimizer and no tokenizer training
> here. It reads weights someone else produced and runs them forward.
>
> And be clear about what the model is. A 10.9-million-parameter character model
> imitates a **style**. It knows no facts. Every sentence it produces is
> invented, including the ones that sound like citations. The program says so
> itself on startup, and that warning is not decoration.

## What is not in this repository

This repository is the Eiffel front-end and nothing else. Deliberately absent,
and excluded by `.gitignore`:

| Absent | What it is |
|--------|------------|
| `model.bin` | the trained weights (about 21 MB) |
| `bin/` | the runnable folder — `RixGPT.exe` beside `model.bin` |
| `installer/src/` | the Inno Setup staging folder, which holds a copy of both |
| `installer/RixGPT-Setup.exe` | the compiled installer (about 21 MB) |

The weights are produced by Larry Rix's private training pipeline, from his own
writing, and are not distributed here. `installer/RixGPT.iss` **is** source and
stays: it documents exactly how the shipped installer is assembled.

Without a `model.bin` the program builds and starts, prints the path it could
not find, and exits. That is the intended behavior, and the compile is the
acceptance test this repository can offer.

## The RIXG weights file

`RIXGPT_MODEL.make` reads a self-describing header, so one binary serves any
model shape. Everything is little-endian, and every integer is 32-bit.

| Bytes | Field |
|-------|-------|
| 4 | magic — the ASCII characters `RIXG` |
| 4 | format version |
| 4 | `dtype`, present only when version ≥ 2: `0` = fp32, `1` = fp16 |
| 5 × 4 | `n_layer`, `n_head`, `n_embd`, `vocab_size`, `block_size` |
| 4 | vocabulary count (equal to `vocab_size`) |
| `vocab_size` × 4 | the Unicode code point of each vocabulary symbol — this *is* the tokenizer |
| rest | the weight tensors, back to back, in serialization order |

There is no tensor-name table. The loader walks the same fixed order the trainer
wrote and records a float offset for each tensor:

1. token embedding `[vocab_size, n_embd]`
2. position embedding `[block_size, n_embd]`
3. for each of `n_layer` blocks, in order: `ln1.weight`, `ln1.bias`, the fused
   QKV projection `[3·n_embd, n_embd]` and its bias, the attention output
   projection `[n_embd, n_embd]` and its bias, `ln2.weight`, `ln2.bias`, the MLP
   `fc1 [4·n_embd, n_embd]` and its bias, and `fc2 [n_embd, 4·n_embd]` and its bias
4. the final layernorm's weight and bias

The output head is not stored separately: the logits are read straight off the
token embedding, so the model ties input and output embeddings.

The shipped `model.bin` declares version 2, `dtype` 1 (fp16), 6 layers, 6 heads,
`n_embd` 384 (head dimension 64), a vocabulary of 316 code points and a context
window of 256 characters — 10,867,200 weights, the 10.9 M of the description.

## How the inference works

- **fp16 by hand.** Eiffel has no binary16 type, so `half_to_double` decodes the
  sign, exponent and mantissa of each 16-bit pattern arithmetically, subnormals
  included, into the `REAL_64` weight buffer read once at load time.
- **One token at a time, with a KV cache.** `forward` handles a single position
  and appends that position's keys and values to `kcache` / `vcache`, both sized
  `n_layer × block_size × n_embd`. Prompt characters are streamed through the
  same path, so a continuation never re-runs attention over text it has already
  seen.
- **Causal attention comes for free.** Because generation is strictly
  left-to-right and the cache only ever holds positions `0 .. pos`, there is no
  mask to build — the loop simply stops at `pos`.
- **GELU without a library.** `erf` is the Abramowitz & Stegun 7.1.26 rational
  approximation (error below 1.5 × 10⁻⁷), which is what makes the exact GELU
  affordable in pure Eiffel.
- **Sampling.** Temperature scaling, then a top-k floor found by partial
  selection sort (the vocabulary is small enough that this is cheaper than a full
  sort), then softmax and an inverse-CDF draw from an xorshift32 stream. The app
  reseeds per turn from the clock, the prompt's hash and the turn number, so the
  same prompt does not give the same continuation twice.
- **Unknown characters.** Anything outside the 316-symbol vocabulary maps to
  index 0 rather than raising.

The engine's numerics were validated against a numpy reference which in turn
matches PyTorch to 6 × 10⁻⁶ on the logits.

## Building

```bash
cd /d/prod/simple_rixgpt
/d/prod/ec.sh release -config simple_rixgpt.ecf -target simple_rixgpt_app
```

`release` finalizes twice and leaves both binaries in
`EIFGENs/simple_rixgpt_app/F_code/`: `simple_rixgpt_lean.exe`, with assertions
compiled out, is the one that ships as `RixGPT.exe`; `simple_rixgpt_dbc.exe`
keeps the contracts in.

The lean executable links against `KERNEL32.dll` only. Nothing needs to be
installed on the machine that runs it.

## Distributable

**A distributable is a folder, not a zip and not a single file.** It holds
exactly two things, side by side:

```
RixGPT\
    RixGPT.exe        (the finalized lean binary, renamed)
    model.bin         (the weights)
```

Run it from a command prompt in that folder:

```bat
cd /d D:\path\to\RixGPT
RixGPT.exe
```

`RixGPT.exe --help` prints the usage screen. The first argument, if given, is the
path to a weights file; with no argument the program looks for `model.bin` in the
current directory, which is why the two files must sit together and why the
folder is the unit you copy.

`installer/RixGPT.iss` builds the same folder as a Windows installer, adding a
`Start RixGPT.bat` wrapper that runs `chcp 65001` first so em-dashes and Hebrew
render in the console. Compile it with Inno Setup 6:

```bat
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" RixGPT.iss
```

That step needs `installer/src/` populated with the executable, the weights, the
batch file and the readme — none of which are in this repository.

## Targets

- `simple_rixgpt` — library target: `RIXGPT_MODEL` alone, usable from another ECF
- `simple_rixgpt_app` — the console executable
- `simple_rixgpt_tests` — a compile proof (see below)

## Testing

```bash
/d/prod/ec.sh test -config simple_rixgpt.ecf -target simple_rixgpt_tests
./EIFGENs/simple_rixgpt_tests/F_code/simple_rixgpt.exe
```

This is a **compile proof, not behavioral coverage**, and the file says so.
`RIXGPT_MODEL` has exactly one creation procedure and it takes a path to a
weights file, so nothing in the class can be exercised without the weights this
repository does not carry. What the target does prove is that both classes
type-check and finalize — which is what breaks first when the engine is edited.

## Layout

| Path | What |
|------|------|
| `src/rixgpt_model.e` | the whole engine: loader, forward pass, attention, sampling |
| `app/rixgpt_app.e` | the console REPL and the help screen |
| `testing/test_app.e` | the compile proof |
| `installer/RixGPT.iss` | Inno Setup script for the Windows installer |
| `docs/` | the documentation site published to GitHub Pages |
| `simple_rixgpt.ecf` | library, app and test targets |

## Dependencies

- `base` (ISE)
- `time` (ISE) — the app seeds its sampler from the clock

Nothing else. No `simple_*` library, no C library, no network.

## License

MIT License — see [LICENSE](LICENSE).

---

Part of the [Simple Eiffel](https://github.com/simple-eiffel) ecosystem.
