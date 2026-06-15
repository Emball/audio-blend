# audio-blend

Perceptual blending of two audio restorations (Apollo neural + algorithmic e.g. Stereo Tool) into a single output that takes the best of each.

## Problem

Apollo neural restoration excels at artifact suppression and frequency reconstruction on heavily compressed files (64–128kbps). At moderate bitrates (192–256kbps), the source is already close to ground truth — Apollo's architecture overcooks it: softens transients, diffuses ambience, tames high-end air. An algorithmic restoration (Stereo Tool) preserves more of the original character but retains compression artifacts. The goal is to blend them adaptively so each source contributes where it's strongest.

## Architecture

Single self-contained script: `blend.py`

### Inputs
- `--apollo` — Apollo-restored WAV
- `--algo` — Algorithmically-restored WAV (Stereo Tool or similar)
- `--output` — Output WAV path
- `--deltawave` — Optional: DeltaWave difference WAV between the two, used to calibrate band weights
- `--crossover` — Optional: manual override Hz for the main crossover (default: auto-detected)
- `--format` — Output format: wav (default), flac, mp3

### Pipeline (all processing in M/S domain, chunk-based for memory efficiency)

1. **Align** — xcorr-based sample-accurate alignment of the two inputs
2. **M/S encode** — process Mid and Side independently (ambience lives in Side)
3. **Band split** — multiband FIR decomposition into 3 regions:
   - Low/mid (below ~8kHz): Apollo typically wins — cleaner, fewer artifacts
   - Transition band (~8–14kHz): blend weighted by per-band spectral confidence
   - Air/ambience (above ~14kHz): algo typically wins — more original character
4. **STFT masking** — patch algo high-frequency content into Apollo where Apollo's energy is below threshold (recovers tamed air without reintroducing low-mid artifacts)
5. **Transient protection** — detect transient regions via envelope follower; in those regions, increase algo weight to preserve attack character
6. **Crossover auto-detection** — if `--deltawave` provided, analyze difference spectrum to find the Hz threshold where the two sources diverge most; otherwise use defaults
7. **Reconstruct** — IFIR recombine + M/S decode
8. **Encode** — write output via ffmpeg (wav passthrough or lossy/lossless encode)

### Key constants (tunable)
- `DEFAULT_CROSSOVER_LOW = 8000` Hz
- `DEFAULT_CROSSOVER_HIGH = 14000` Hz
- `STFT_THRESHOLD_DB = -118.0`
- `TRANSIENT_THRESHOLD = 0.35` (ratio of peak to RMS)
- `CHUNK_SIZE = 262144` samples

## Dependencies
- numpy, scipy — DSP
- soundfile — audio I/O
- ffmpeg (system) — encoding

## Versioning
MAJOR.MINOR.PATCH.MICRO — bump thresholds: 300+ lines → MAJOR, 100+ → MINOR, 20+ → PATCH, 1+ → MICRO

## Version history
- 0.0.0.1 — initial repo + AGENTS.md
