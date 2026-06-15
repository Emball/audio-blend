#!/usr/bin/env python3
"""
blend.py — Perceptual blending of Apollo neural + algorithmic audio restorations.

5-band crossover via LP subtraction — all bands share identical group delay,
eliminating the discontinuities caused by cascaded FIR bandpass paths.

Crossover architecture calibrated from DeltaWave analysis (MP3 vs Apollo,
MP3 vs ST, Apollo vs ST) on 192kbps MP3 material, June 2026.

Usage:
    python blend.py --apollo apollo.wav --algo stereo_tool.wav --output blended.wav
    python blend.py --apollo apollo.wav --algo stereo_tool.wav --output blended.wav --format flac
    python blend.py --apollo apollo.wav --algo stereo_tool.wav --output blended.wav --crossover 13000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import firwin, fftconvolve
from scipy.fft import next_fast_len

# ---------------------------------------------------------------------------
# Constants — calibrated from DeltaWave measurements (2026-06-14)
#
# Band structure (MP3-centric):
#   LOW   0 – XOVER_PHASE_KNEE : ST wins — 1.81° phase vs Apollo's 15.86°
#   MID   XOVER_PHASE_KNEE – XOVER_MID_TOP : Apollo artifact cleanup
#   AIR   XOVER_MID_TOP – XOVER_APOLLO_WALL : Apollo +8dB (Delta of Spectra)
#   ULTRA XOVER_APOLLO_WALL – XOVER_ST_WALL : both synthesizing, even blend
#   SILK  XOVER_ST_WALL – Nyq : ST conservative preferred
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
CHANNELS    = 2
CHUNK_SIZE  = 262144  # ~5.9s at 44100

XOVER_PHASE_KNEE  = 10700   # Hz
XOVER_MID_TOP     = 13000   # Hz
XOVER_APOLLO_WALL = 15500   # Hz
XOVER_ST_WALL     = 16500   # Hz

# Per-band Apollo weights (ST weight = 1 - w)
W_LOW   = 0.35
W_MID   = 0.60
W_AIR   = 0.65
W_ULTRA = 0.50
W_SILK  = 0.40

TRANSIENT_THRESHOLD  = 0.35
TRANSIENT_ALGO_BOOST = 0.15  # extra ST weight during transients

FIR_TAPS = 2049  # must be odd; group delay = (FIR_TAPS - 1) / 2 = 1024 samples

FORMAT_DEFAULTS = {
    "wav":  {"ext": "wav",  "codec": None},
    "flac": {"ext": "flac", "codec": "flac", "default_bit_depth": 24},
    "mp3":  {"ext": "mp3",  "codec": "libmp3lame", "default_bitrate": "320k"},
}


# ---------------------------------------------------------------------------
# FIR design — linear-phase, fixed tap count so all filters share same group delay
# ---------------------------------------------------------------------------
def _design_lp(freq: float, fs: int) -> np.ndarray:
    nyq  = fs / 2.0
    norm = np.clip(freq / nyq, 0.001, 0.999)
    h = firwin(FIR_TAPS, norm, window="hamming", pass_zero=True)
    return h.astype(np.float32)


# ---------------------------------------------------------------------------
# Streaming FIR (overlap-save)
# ---------------------------------------------------------------------------
class StreamingFIR:
    def __init__(self, taps: np.ndarray):
        self.h      = taps.astype(np.float32)
        self.L      = len(taps)
        self._H     = None
        self._fft_n = 0
        self._tail  = None

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
        tail_len   = self.L - 1
        self._tail = (chunk[-tail_len:] if M >= tail_len
                      else np.concatenate([self._tail[M:], chunk])).copy()
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
    n      = min(len(mono_a), len(mono_b), fs * 10)
    ref    = mono_b[:n] - mono_b[:n].mean()
    test   = mono_a[:n] - mono_a[:n].mean()
    corr   = fftconvolve(ref, test[::-1], mode='full')
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
# Pipeline state
#
# Band extraction via LP subtraction — every band passes through exactly ONE
# FIR filter, so group delay is identical across all bands. No cascading.
#
# lp1 = LP @ XOVER_PHASE_KNEE
# lp2 = LP @ XOVER_MID_TOP
# lp3 = LP @ XOVER_APOLLO_WALL
# lp4 = LP @ XOVER_ST_WALL
#
# band1 (LOW)   = lp1(x)
# band2 (MID)   = lp2(x) - lp1(x)
# band3 (AIR)   = lp3(x) - lp2(x)
# band4 (ULTRA) = lp4(x) - lp3(x)
# band5 (SILK)  = x_delayed - lp4(x)     ← needs matching delay on raw signal
#
# All lp filters use FIR_TAPS taps → group delay = (FIR_TAPS-1)/2 samples.
# The raw signal (band5 complement) is delayed by the same amount via StreamingDelay.
# ---------------------------------------------------------------------------
GROUP_DELAY = (FIR_TAPS - 1) // 2  # samples


class StreamingDelay:
    def __init__(self, D: int):
        self.D    = D
        self._buf = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        M, C = chunk.shape
        if self._buf is None:
            self._buf = np.zeros((self.D, C), dtype=np.float32)
        combined  = np.concatenate([self._buf, chunk])
        out        = combined[:M].copy()
        self._buf  = combined[M: M + self.D].copy()
        return out


class BlendState:
    def __init__(self):
        # Four LP filters — one per crossover point, same tap count = same group delay
        self.lp1_a = StreamingFIR(_design_lp(XOVER_PHASE_KNEE,  SAMPLE_RATE))
        self.lp1_b = StreamingFIR(_design_lp(XOVER_PHASE_KNEE,  SAMPLE_RATE))
        self.lp2_a = StreamingFIR(_design_lp(XOVER_MID_TOP,     SAMPLE_RATE))
        self.lp2_b = StreamingFIR(_design_lp(XOVER_MID_TOP,     SAMPLE_RATE))
        self.lp3_a = StreamingFIR(_design_lp(XOVER_APOLLO_WALL, SAMPLE_RATE))
        self.lp3_b = StreamingFIR(_design_lp(XOVER_APOLLO_WALL, SAMPLE_RATE))
        self.lp4_a = StreamingFIR(_design_lp(XOVER_ST_WALL,     SAMPLE_RATE))
        self.lp4_b = StreamingFIR(_design_lp(XOVER_ST_WALL,     SAMPLE_RATE))

        # Delay raw signal to match FIR group delay for band5 complement
        self.delay_a = StreamingDelay(GROUP_DELAY)
        self.delay_b = StreamingDelay(GROUP_DELAY)

        self.total_delay = GROUP_DELAY


# ---------------------------------------------------------------------------
# Process one chunk (M/S domain)
# ---------------------------------------------------------------------------
def process_chunk(apollo_ms: np.ndarray, algo_ms: np.ndarray,
                  st: BlendState) -> np.ndarray:

    # Compute all LP outputs — each signal goes through exactly one filter
    lp1_a = st.lp1_a.process(apollo_ms)
    lp1_b = st.lp1_b.process(algo_ms)
    lp2_a = st.lp2_a.process(apollo_ms)
    lp2_b = st.lp2_b.process(algo_ms)
    lp3_a = st.lp3_a.process(apollo_ms)
    lp3_b = st.lp3_b.process(algo_ms)
    lp4_a = st.lp4_a.process(apollo_ms)
    lp4_b = st.lp4_b.process(algo_ms)

    # Delay raw signal to match group delay for SILK complement
    raw_a = st.delay_a.process(apollo_ms)
    raw_b = st.delay_b.process(algo_ms)

    # Extract bands via subtraction — uniform group delay across all bands
    b1_a = lp1_a                   # LOW   apollo
    b1_b = lp1_b                   # LOW   algo
    b2_a = lp2_a - lp1_a           # MID   apollo
    b2_b = lp2_b - lp1_b           # MID   algo
    b3_a = lp3_a - lp2_a           # AIR   apollo
    b3_b = lp3_b - lp2_b           # AIR   algo
    b4_a = lp4_a - lp3_a           # ULTRA apollo
    b4_b = lp4_b - lp3_b           # ULTRA algo
    b5_a = raw_a - lp4_a           # SILK  apollo
    b5_b = raw_b - lp4_b           # SILK  algo

    # Per-band weights
    t_str = transient_strength(apollo_ms)
    w1 = np.float32(W_LOW   - t_str * TRANSIENT_ALGO_BOOST)
    w2 = np.float32(W_MID)
    w3 = np.float32(W_AIR)
    w4 = np.float32(W_ULTRA)
    w5 = np.float32(W_SILK)

    out = (b1_a * w1           + b1_b * (1.0 - w1) +
           b2_a * w2           + b2_b * (1.0 - w2) +
           b3_a * w3           + b3_b * (1.0 - w3) +
           b4_a * w4           + b4_b * (1.0 - w4) +
           b5_a * w5           + b5_b * (1.0 - w5))
    return out.astype(np.float32)


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
    parser.add_argument('--apollo',     required=True,  help='Apollo-restored WAV')
    parser.add_argument('--algo',       required=True,  help='Algorithmic restoration WAV')
    parser.add_argument('--output',     required=True,  help='Output file path')
    parser.add_argument('--format',     default='wav',  choices=list(FORMAT_DEFAULTS))
    parser.add_argument('--bitrate',    default=None,   help='Bitrate for lossy output')
    parser.add_argument('--bit-depth',  default=None,   type=int, help='Bit depth for FLAC')
    parser.add_argument('--chunk-size', default=CHUNK_SIZE, type=int)
    parser.add_argument('--crossover',  default=None,   type=float,
                        help='Shift primary crossover point (XOVER_MID_TOP default 13000Hz)')
    args = parser.parse_args()

    if args.crossover is not None:
        global XOVER_MID_TOP, XOVER_APOLLO_WALL, XOVER_ST_WALL, XOVER_PHASE_KNEE
        shift             = args.crossover - XOVER_MID_TOP
        XOVER_PHASE_KNEE  = max(1000, XOVER_PHASE_KNEE  + shift)
        XOVER_MID_TOP     = args.crossover
        XOVER_APOLLO_WALL = max(XOVER_MID_TOP + 500,   XOVER_APOLLO_WALL + shift)
        XOVER_ST_WALL     = max(XOVER_APOLLO_WALL + 500, XOVER_ST_WALL   + shift)

    print(f"[blend] 5-band crossovers: "
          f"{XOVER_PHASE_KNEE} / {XOVER_MID_TOP} / {XOVER_APOLLO_WALL} / {XOVER_ST_WALL} Hz",
          file=sys.stderr)
    print(f"[blend] band weights (Apollo): "
          f"LOW={W_LOW} MID={W_MID} AIR={W_AIR} ULTRA={W_ULTRA} SILK={W_SILK}",
          file=sys.stderr)

    apollo_path = Path(args.apollo)
    algo_path   = Path(args.algo)
    output_path = Path(args.output)

    print(f"[blend] loading {apollo_path.name}...", file=sys.stderr)
    apollo = load_audio(apollo_path, SAMPLE_RATE)
    print(f"[blend] loading {algo_path.name}...", file=sys.stderr)
    algo   = load_audio(algo_path,   SAMPLE_RATE)

    print(f"[blend] aligning sources...", file=sys.stderr)
    apollo, algo = align_sources(apollo, algo, SAMPLE_RATE)
    total_samples = len(apollo)
    print(f"[blend] {total_samples / SAMPLE_RATE:.1f}s to process", file=sys.stderr)

    st         = BlendState()
    chunks_out = []
    skip       = st.total_delay
    chunk_size = args.chunk_size

    for i in range(0, total_samples, chunk_size):
        a_chunk = apollo[i: i + chunk_size]
        b_chunk = algo[i:   i + chunk_size]
        n = min(len(a_chunk), len(b_chunk))
        a_chunk, b_chunk = a_chunk[:n], b_chunk[:n]

        a_ms   = ms_encode(a_chunk)
        b_ms   = ms_encode(b_chunk)
        out_ms = process_chunk(a_ms, b_ms, st)
        out    = ms_decode(out_ms)

        if skip > 0:
            if out.shape[0] <= skip:
                skip -= out.shape[0]
                continue
            out  = out[skip:]
            skip = 0

        chunks_out.append(out)
        pct = min(100, i * 100 // total_samples)
        print(f"\r  {pct:3d}% — {i / SAMPLE_RATE:.1f}s / {total_samples / SAMPLE_RATE:.1f}s",
              end='', file=sys.stderr)

    # Flush group delay
    flush_in  = np.zeros((GROUP_DELAY, CHANNELS), dtype=np.float32)
    flush_ms  = process_chunk(ms_encode(flush_in), ms_encode(flush_in), st)
    flush_out = ms_decode(flush_ms)
    chunks_out.append(flush_out[:GROUP_DELAY])

    print(f"\r  done — {total_samples / SAMPLE_RATE:.1f}s processed" + " " * 20, file=sys.stderr)

    result = np.concatenate(chunks_out, axis=0)[:total_samples]
    result = np.clip(result, -1.0, 1.0)

    print(f"[blend] writing {output_path}...", file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_output(result, output_path, args.format,
                 bitrate=args.bitrate, bit_depth=args.bit_depth)
    print(f"[blend] done → {output_path}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
