#!/usr/bin/env python3
"""
blend.py — Perceptual blending of Apollo neural + algorithmic audio restorations.

Crossover architecture calibrated from DeltaWave analysis (MP3 vs Apollo,
MP3 vs ST, Apollo vs ST) on 192kbps MP3 material, June 2026.

Usage:
    python blend.py --apollo apollo.wav --algo stereo_tool.wav --output blended.wav
    python blend.py --apollo apollo.wav --algo stereo_tool.wav --output blended.wav --format flac
    python blend.py --apollo apollo.wav --algo stereo_tool.wav --output blended.wav --crossover 13000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import firwin, fftconvolve
from scipy.fft import next_fast_len
from scipy.signal import stft as scipy_stft, istft as scipy_istft

# ---------------------------------------------------------------------------
# Constants — calibrated from DeltaWave measurements (2026-06-14)
#
# Band structure (MP3-centric):
#   LOW   0 – XOVER_PHASE_KNEE : Apollo and ST nearly identical spectrally.
#                                  ST wins: 1.81° phase error vs Apollo's 15.86°
#                                  → blend 35/65 Apollo/ST (artifact cleanup / phase)
#   MID   XOVER_PHASE_KNEE – XOVER_MID_TOP : gradual divergence begins.
#                                  Apollo has more controlled energy.
#                                  → blend 60/40 Apollo/ST
#   AIR   XOVER_MID_TOP – XOVER_APOLLO_WALL : Apollo +8dB vs ST, synthesis starting.
#                                  Apollo wins on energy, temper with ST for continuity.
#                                  → blend 65/35 Apollo/ST
#   ULTRA XOVER_APOLLO_WALL – XOVER_ST_WALL : Apollo brickwall synthesis at +18dB.
#                                  Both are synthesizing. Apollo brighter, ST conservative.
#                                  → blend 50/50
#   SILK  XOVER_ST_WALL – Nyq   : Both in extreme synthesis territory. Conservative blend.
#                                  → blend 40/60 Apollo/ST
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
CHANNELS    = 2
CHUNK_SIZE  = 262144  # ~5.9s at 44100

XOVER_PHASE_KNEE  = 10700   # Hz: where Apollo/ST phase starts diverging (15.86° avg)
XOVER_MID_TOP     = 13000   # Hz: Delta of Spectra rising above 2dB; red on spectrogram
XOVER_APOLLO_WALL = 15500   # Hz: Apollo's synthesis brickwall (15.8kHz from MP3 vs Apollo)
XOVER_ST_WALL     = 16500   # Hz: ST's roll-off point (16.5kHz from MP3 vs ST)

# Per-band Apollo weights (ST weight = 1 - apollo_w)
W_LOW   = 0.35   # ST phase advantage dominates
W_MID   = 0.60   # Apollo artifact cleanup
W_AIR   = 0.65   # Apollo energy advantage
W_ULTRA = 0.50   # both synthesizing, blend evenly
W_SILK  = 0.40   # ST more conservative above its own wall

TRANSIENT_THRESHOLD  = 0.35   # peak/RMS ratio above which transient weighting activates
TRANSIENT_ALGO_BOOST = 0.15   # extra ST weight during transients (phase coherence)

FORMAT_DEFAULTS = {
    "wav":  {"ext": "wav",  "codec": None},
    "flac": {"ext": "flac", "codec": "flac", "default_bit_depth": 24},
    "mp3":  {"ext": "mp3",  "codec": "libmp3lame", "default_bitrate": "320k"},
}


# ---------------------------------------------------------------------------
# FIR design — linear-phase, odd-tap, Hamming window
# ---------------------------------------------------------------------------
def _design_fir(pass_type: str, freq: float, fs: int, base_taps: int = 513) -> np.ndarray:
    nyq = fs / 2.0
    norm = np.clip(freq / nyq, 0.001, 0.999)
    numtaps = (base_taps * 4) | 1  # force odd
    if pass_type == "lp":
        h = firwin(numtaps, norm, window="hamming", pass_zero=True)
    elif pass_type == "hp":
        h = firwin(numtaps, norm, window="hamming", pass_zero=False)
    else:
        raise ValueError(f"Unknown pass_type: {pass_type}")
    return h.astype(np.float32)


# ---------------------------------------------------------------------------
# Streaming FIR (overlap-save)
# ---------------------------------------------------------------------------
class StreamingFIR:
    def __init__(self, taps: np.ndarray):
        self.h = taps.astype(np.float32)
        self.L = len(taps)
        self._H    = None
        self._fft_n = 0
        self._tail = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        M, C = chunk.shape
        if self._tail is None:
            self._tail  = np.zeros((self.L - 1, C), dtype=np.float32)
            self._fft_n = next_fast_len(M + self.L - 1)
            self._H     = np.fft.rfft(self.h, self._fft_n)
        out = np.empty((M, C), dtype=np.float32)
        for c in range(C):
            block = np.concatenate([self._tail[:, c], chunk[:, c]])
            Y = np.fft.rfft(block, self._fft_n) * self._H
            y = np.fft.irfft(Y, self._fft_n)
            out[:, c] = y[self.L - 1: self.L - 1 + M]
        tail_len = self.L - 1
        self._tail = (chunk[-tail_len:] if M >= tail_len
                      else np.concatenate([self._tail[M:], chunk])).copy()
        return out


class StreamingDelay:
    def __init__(self, D: int):
        self.D    = D
        self._buf = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        M, C = chunk.shape
        if self._buf is None:
            self._buf = np.zeros((self.D, C), dtype=np.float32)
        combined = np.concatenate([self._buf, chunk])
        out       = combined[:M].copy()
        self._buf = combined[M: M + self.D].copy()
        return out


# ---------------------------------------------------------------------------
# M/S encode / decode
# ---------------------------------------------------------------------------
def ms_encode(s: np.ndarray) -> np.ndarray:
    return np.column_stack(((s[:, 0] + s[:, 1]) * 0.5,
                             (s[:, 0] - s[:, 1]) * 0.5))

def ms_decode(ms: np.ndarray) -> np.ndarray:
    return np.column_stack((ms[:, 0] + ms[:, 1],
                             ms[:, 0] - ms[:, 1]))


# ---------------------------------------------------------------------------
# Transient detection
# ---------------------------------------------------------------------------
def transient_strength(chunk: np.ndarray, threshold: float = TRANSIENT_THRESHOLD) -> float:
    rms  = np.sqrt(np.mean(chunk ** 2) + 1e-12)
    peak = np.max(np.abs(chunk))
    ratio = peak / (rms + 1e-12)
    return float(np.clip((ratio / (1.0 / threshold) - 1.0), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def align_sources(apollo: np.ndarray, algo: np.ndarray, fs: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    MAX_OFFSET = fs // 4
    mono_a = apollo[:, 0].astype(np.float64)
    mono_b = algo[:, 0].astype(np.float64)
    n = min(len(mono_a), len(mono_b), fs * 10)
    ref  = mono_b[:n] - mono_b[:n].mean()
    test = mono_a[:n] - mono_a[:n].mean()
    corr = fftconvolve(ref, test[::-1], mode='full')
    mid    = len(test) - 1
    search = corr[mid - MAX_OFFSET: mid + MAX_OFFSET + 1]
    offset = MAX_OFFSET - int(np.argmax(np.abs(search)))
    if offset != 0:
        print(f"[align] correcting {offset:+d} samples", file=sys.stderr)
    if offset > 0:
        apollo = apollo[offset:]
    elif offset < 0:
        algo = algo[-offset:]
    min_len = min(len(apollo), len(algo))
    return apollo[:min_len], algo[:min_len]


# ---------------------------------------------------------------------------
# Pipeline state — 5-band crossover network
# ---------------------------------------------------------------------------
class BlendState:
    def __init__(self):
        # FIR group delay: (numtaps - 1) / 2 = (513*4 | 1 - 1) / 2 = 1024 samples
        self._D = ((513 * 4) | 1 - 1) // 2

        # Band 1: low-pass below XOVER_PHASE_KNEE
        self.lp1_a = StreamingFIR(_design_fir("lp", XOVER_PHASE_KNEE, SAMPLE_RATE))
        self.lp1_b = StreamingFIR(_design_fir("lp", XOVER_PHASE_KNEE, SAMPLE_RATE))

        # Band 2: band-pass XOVER_PHASE_KNEE – XOVER_MID_TOP
        self.bp2_hp_a = StreamingFIR(_design_fir("hp", XOVER_PHASE_KNEE, SAMPLE_RATE))
        self.bp2_hp_b = StreamingFIR(_design_fir("hp", XOVER_PHASE_KNEE, SAMPLE_RATE))
        self.bp2_lp_a = StreamingFIR(_design_fir("lp", XOVER_MID_TOP,   SAMPLE_RATE))
        self.bp2_lp_b = StreamingFIR(_design_fir("lp", XOVER_MID_TOP,   SAMPLE_RATE))

        # Band 3: band-pass XOVER_MID_TOP – XOVER_APOLLO_WALL
        self.bp3_hp_a = StreamingFIR(_design_fir("hp", XOVER_MID_TOP,     SAMPLE_RATE))
        self.bp3_hp_b = StreamingFIR(_design_fir("hp", XOVER_MID_TOP,     SAMPLE_RATE))
        self.bp3_lp_a = StreamingFIR(_design_fir("lp", XOVER_APOLLO_WALL, SAMPLE_RATE))
        self.bp3_lp_b = StreamingFIR(_design_fir("lp", XOVER_APOLLO_WALL, SAMPLE_RATE))

        # Band 4: band-pass XOVER_APOLLO_WALL – XOVER_ST_WALL
        self.bp4_hp_a = StreamingFIR(_design_fir("hp", XOVER_APOLLO_WALL, SAMPLE_RATE))
        self.bp4_hp_b = StreamingFIR(_design_fir("hp", XOVER_APOLLO_WALL, SAMPLE_RATE))
        self.bp4_lp_a = StreamingFIR(_design_fir("lp", XOVER_ST_WALL,    SAMPLE_RATE))
        self.bp4_lp_b = StreamingFIR(_design_fir("lp", XOVER_ST_WALL,    SAMPLE_RATE))

        # Band 5: high-pass above XOVER_ST_WALL
        self.hp5_a = StreamingFIR(_design_fir("hp", XOVER_ST_WALL, SAMPLE_RATE))
        self.hp5_b = StreamingFIR(_design_fir("hp", XOVER_ST_WALL, SAMPLE_RATE))

        # Delay passthrough (time-align un-filtered path — not used in blend but
        # kept so group delay on all bands is equal at reconstruction)
        self.total_delay = self._D * 2  # two cascaded filters in band-pass paths


# ---------------------------------------------------------------------------
# Process one chunk (M/S domain)
# ---------------------------------------------------------------------------
def process_chunk(apollo_ms: np.ndarray, algo_ms: np.ndarray,
                  st: BlendState) -> np.ndarray:

    # Band 1: LOW (0 – XOVER_PHASE_KNEE) — ST phase advantage
    b1_a = st.lp1_a.process(apollo_ms)
    b1_b = st.lp1_b.process(algo_ms)
    t_str = transient_strength(apollo_ms)
    # During transients, lean further toward ST for phase coherence
    w_a1 = np.float32(W_LOW  - t_str * TRANSIENT_ALGO_BOOST)
    b1   = b1_a * w_a1 + b1_b * np.float32(1.0 - w_a1)

    # Band 2: MID (XOVER_PHASE_KNEE – XOVER_MID_TOP) — Apollo artifact cleanup
    b2_a = st.bp2_lp_a.process(st.bp2_hp_a.process(apollo_ms))
    b2_b = st.bp2_lp_b.process(st.bp2_hp_b.process(algo_ms))
    b2   = b2_a * np.float32(W_MID) + b2_b * np.float32(1.0 - W_MID)

    # Band 3: AIR (XOVER_MID_TOP – XOVER_APOLLO_WALL) — Apollo energy (+8dB)
    b3_a = st.bp3_lp_a.process(st.bp3_hp_a.process(apollo_ms))
    b3_b = st.bp3_lp_b.process(st.bp3_hp_b.process(algo_ms))
    b3   = b3_a * np.float32(W_AIR) + b3_b * np.float32(1.0 - W_AIR)

    # Band 4: ULTRA (XOVER_APOLLO_WALL – XOVER_ST_WALL) — even blend
    b4_a = st.bp4_lp_a.process(st.bp4_hp_a.process(apollo_ms))
    b4_b = st.bp4_lp_b.process(st.bp4_hp_b.process(algo_ms))
    b4   = b4_a * np.float32(W_ULTRA) + b4_b * np.float32(1.0 - W_ULTRA)

    # Band 5: SILK (XOVER_ST_WALL – Nyq) — ST more conservative above its wall
    b5_a = st.hp5_a.process(apollo_ms)
    b5_b = st.hp5_b.process(algo_ms)
    b5   = b5_a * np.float32(W_SILK) + b5_b * np.float32(1.0 - W_SILK)

    return b1 + b2 + b3 + b4 + b5


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def load_audio(path: Path, target_sr: int) -> np.ndarray:
    data, sr = sf.read(str(path), always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    if data.shape[1] > 2:
        data = data[:, :2]
    if sr != target_sr:
        print(f"[load] {path.name}: resampling {sr} → {target_sr}", file=sys.stderr)
        import os
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, data, sr, subtype='FLOAT')
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as out:
            out_path = out.name
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                        '-i', tmp_path, '-ar', str(target_sr), '-y', out_path], check=True)
        data, _ = sf.read(out_path, always_2d=True)
        os.unlink(tmp_path); os.unlink(out_path)
    return data.astype(np.float32)


def write_output(data: np.ndarray, output_path: Path, format_name: str,
                 bitrate: str | None = None, bit_depth: int | None = None) -> None:
    fmt = FORMAT_DEFAULTS[format_name]
    if format_name == 'wav':
        sf.write(str(output_path), data, SAMPLE_RATE, subtype='FLOAT')
        return
    import os
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, data, SAMPLE_RATE, subtype='FLOAT')
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error',
           '-i', tmp_path, '-c:a', fmt['codec']]
    br = bitrate or fmt.get('default_bitrate')
    bd = bit_depth or fmt.get('default_bit_depth')
    if br:
        cmd.extend(['-b:a', br])
    if bd and format_name == 'flac':
        cmd.extend(['-sample_fmt', f's{bd}'])
    cmd.extend(['-y', str(output_path)])
    subprocess.run(cmd, check=True)
    os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Blend Apollo + algorithmic restorations (MP3-tuned)")
    parser.add_argument('--apollo',    required=True, help='Apollo-restored WAV')
    parser.add_argument('--algo',      required=True, help='Algorithmic restoration WAV (Stereo Tool etc.)')
    parser.add_argument('--output',    required=True, help='Output file path')
    parser.add_argument('--format',    default='wav', choices=list(FORMAT_DEFAULTS))
    parser.add_argument('--bitrate',   default=None,  help='Bitrate for lossy output')
    parser.add_argument('--bit-depth', default=None,  type=int, help='Bit depth for FLAC')
    parser.add_argument('--chunk-size',default=CHUNK_SIZE, type=int)
    parser.add_argument('--crossover', default=None, type=float,
                        help='Override the primary crossover point (XOVER_MID_TOP, default 13000)')
    args = parser.parse_args()

    apollo_path = Path(args.apollo)
    algo_path   = Path(args.algo)
    output_path = Path(args.output)

    if args.crossover is not None:
        global XOVER_MID_TOP, XOVER_APOLLO_WALL, XOVER_ST_WALL, XOVER_PHASE_KNEE
        shift = args.crossover - XOVER_MID_TOP
        XOVER_MID_TOP     = args.crossover
        XOVER_APOLLO_WALL = max(XOVER_MID_TOP + 500,  XOVER_APOLLO_WALL + shift)
        XOVER_ST_WALL     = max(XOVER_APOLLO_WALL + 500, XOVER_ST_WALL  + shift)
        XOVER_PHASE_KNEE  = max(1000, XOVER_PHASE_KNEE + shift)
        print(f"[blend] crossover override → "
              f"{XOVER_PHASE_KNEE:.0f} / {XOVER_MID_TOP:.0f} / "
              f"{XOVER_APOLLO_WALL:.0f} / {XOVER_ST_WALL:.0f} Hz", file=sys.stderr)
    else:
        print(f"[blend] 5-band crossovers: "
              f"{XOVER_PHASE_KNEE} / {XOVER_MID_TOP} / "
              f"{XOVER_APOLLO_WALL} / {XOVER_ST_WALL} Hz", file=sys.stderr)
        print(f"[blend] band weights (Apollo): "
              f"LOW={W_LOW} MID={W_MID} AIR={W_AIR} ULTRA={W_ULTRA} SILK={W_SILK}",
              file=sys.stderr)

    print(f"[blend] loading {apollo_path.name}...", file=sys.stderr)
    apollo = load_audio(apollo_path, SAMPLE_RATE)
    print(f"[blend] loading {algo_path.name}...", file=sys.stderr)
    algo   = load_audio(algo_path,   SAMPLE_RATE)

    print(f"[blend] aligning sources...", file=sys.stderr)
    apollo, algo = align_sources(apollo, algo, SAMPLE_RATE)
    total_samples = len(apollo)
    print(f"[blend] {total_samples / SAMPLE_RATE:.1f}s to process", file=sys.stderr)

    st = BlendState()
    output_chunks = []
    chunk_size = args.chunk_size

    for i in range(0, total_samples, chunk_size):
        a_chunk = apollo[i: i + chunk_size]
        b_chunk = algo[i:   i + chunk_size]
        n = min(len(a_chunk), len(b_chunk))
        a_chunk, b_chunk = a_chunk[:n], b_chunk[:n]

        # Pad to 1024 boundary
        rem = n % 1024
        if rem:
            pad = 1024 - rem
            a_chunk = np.pad(a_chunk, ((0, pad), (0, 0)))
            b_chunk = np.pad(b_chunk, ((0, pad), (0, 0)))

        a_ms = ms_encode(a_chunk)
        b_ms = ms_encode(b_chunk)
        out_ms = process_chunk(a_ms, b_ms, st)
        out = ms_decode(out_ms)

        output_chunks.append(out[:n])

        pct = min(100, i * 100 // total_samples)
        print(f"\r  {pct:3d}% — {i / SAMPLE_RATE:.1f}s / {total_samples / SAMPLE_RATE:.1f}s",
              end='', file=sys.stderr)

    print(f"\r  done — {total_samples / SAMPLE_RATE:.1f}s processed" + " " * 20, file=sys.stderr)

    result = np.concatenate(output_chunks, axis=0)
    result = np.clip(result, -1.0, 1.0)

    print(f"[blend] writing {output_path}...", file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_output(result, output_path, args.format,
                 bitrate=args.bitrate, bit_depth=args.bit_depth)
    print(f"[blend] done → {output_path}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
