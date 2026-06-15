# audio-blend

Perceptual blending of two audio restorations (Apollo neural + algorithmic e.g. Stereo Tool) into a single output that takes the best of each.

## Problem

Apollo neural restoration excels at artifact suppression on heavily compressed files (64–128kbps). At moderate bitrates (192–256kbps), Apollo overcooks the source: its high-frequency synthesis above ~15.8kHz diverges dramatically from the original, and its STFT/ICB processing introduces phase shift even below 10kHz. Stereo Tool is phase-coherent (1.81° error vs Apollo's 15.86° below 10kHz) and conservative — it barely touches the signal below its own 16.5kHz roll-off. The blend exploits this complementarity.

## Architecture

Single self-contained script: `blend.py`

### Inputs
- `--apollo` — Apollo-restored WAV
- `--algo` — Algorithmic restoration WAV (Stereo Tool or similar)
- `--output` — Output file path
- `--format` — Output format: wav (default), flac, mp3
- `--crossover` — Optional: shift the XOVER_MID_TOP point (13kHz default); all other crossovers shift proportionally
- `--bitrate`, `--bit-depth` — Encoding options

### 5-Band Crossover Pipeline (all in M/S domain, chunk-based)

Calibrated from DeltaWave measurements (MP3 vs Apollo, MP3 vs ST, Apollo vs ST), June 2026, on 192kbps MP3 material.

| Band | Range | Apollo weight | Rationale |
|------|-------|---------------|-----------|
| LOW  | 0 – 10.7kHz | 0.35 | ST wins: phase coherent (1.81°). Apollo adds 15.86° phase error |
| MID  | 10.7 – 13kHz | 0.60 | Apollo wins: better artifact cleanup, divergence beginning |
| AIR  | 13 – 15.5kHz | 0.65 | Apollo wins: +8dB vs ST (Delta of Spectra). ST rolling off |
| ULTRA| 15.5 – 16.5kHz | 0.50 | Both synthesizing. Apollo's brickwall starts, even blend |
| SILK | 16.5kHz – Nyq | 0.40 | ST conservative trim vs Apollo aggressive synthesis. ST preferred |

During transients: LOW band Apollo weight decreases by `TRANSIENT_ALGO_BOOST` (0.15) to preserve ST phase coherence on attacks.

### Pipeline steps
1. Align — xcorr sample-accurate alignment (0.25s search window)
2. M/S encode — Mid and Side processed independently
3. 5-band FIR split — linear-phase Hamming-window FIRs, 2049 taps
4. Per-band blend — calibrated Apollo/ST weights per above table
5. Reconstruct — sum bands, M/S decode
6. Clip and write

### Key constants
- `XOVER_PHASE_KNEE = 10700` Hz
- `XOVER_MID_TOP = 13000` Hz
- `XOVER_APOLLO_WALL = 15500` Hz
- `XOVER_ST_WALL = 16500` Hz
- `CHUNK_SIZE = 262144` samples

## Dependencies
- numpy, scipy — DSP
- soundfile — audio I/O
- ffmpeg (system) — encoding

## DeltaWave calibration data (summary)
- Apollo vs ST, null depth: 39.7dB, difference RMS: -36.45dB
- Phase 0–10kHz: ST=1.81°, Apollo=15.86° — ST is phase-transparent below 10kHz
- Apollo's brickwall synthesis begins at ~15.8kHz (from MP3 vs Apollo: Delta of Spectra cliff)
- ST's roll-off begins at ~16.5kHz (from MP3 vs ST: Delta of Spectra cliff)
- Apollo vs ST Delta of Spectra: 0dB to 10.7kHz, rising to +8dB at 15kHz, +18dB at 21kHz

## Versioning
MAJOR.MINOR.PATCH.MICRO — bump: 300+ lines → MAJOR, 100+ → MINOR, 20+ → PATCH, 1+ → MICRO

## Version history
- 0.0.0.1 — initial repo + AGENTS.md
- 0.1.0.0 — initial blend.py (guessed crossovers: 8kHz/14kHz, 3-band)
- 0.2.0.0 — full rewrite: 5-band DeltaWave-calibrated architecture, removed runtime deltawave flag
