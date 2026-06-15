#!/usr/bin/env python3
"""
blend.py — Perceptual blending of Apollo neural + algorithmic audio restorations.

Usage:
    python blend.py --apollo apollo.wav --algo stereo_tool.wav --output blended.wav
    python blend.py --apollo apollo.wav --algo stereo_tool.wav --output blended.wav --deltawave diff.wav
    python blend.py --apollo apollo.wav --algo stereo_tool.wav --output blended.wav --crossover 11000
    python blend.py --apollo apollo.wav --algo stereo_tool.wav --output blended.wav --format flac
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
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
CHANNELS = 2
CHUNK_SIZE = 262144  # ~5.9s at 44100

DEFAULT_CROSSOVER_LOW  = 8000   # Hz — Apollo preferred below this
DEFAULT_CROSSOVER_HIGH = 14000  # Hz — algo preferred above this
STFT_THRESHOLD_DB      = -118.0
TRANSIENT_THRESHOLD    = 0.35   # peak/RMS ratio above which transient weighting kicks in
TRANSIENT_ALGO_BOOST   = 0.25   # extra algo weight during transients

FORMAT_DEFAULTS = {
    "wav":  {"ext": "wav",  "codec": None},
    "flac": {"ext": "flac", "codec": "flac", "default_bit_depth": 24},
    "mp3":  {"ext": "mp3",  "codec": "libmp3lame", "default_bitrate": "320k"},
}

# ---------------------------------------------------------------------------
# FIR design
# ---------------------------------------------------------------------------
def _design_fir(pass_type: str, freq: float, fs: int, taps: int = 513) -> np.ndarray:
    nyq = fs / 2.0
    norm = freq / nyq
    numtaps = max(taps * 4, 2049) | 1  # ensure odd
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
        self._H = None
        self._fft_n = 0
        self._tail = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        M, C = chunk.shape
        if self._tail is None:
            self._tail = np.zeros((self.L - 1, C), dtype=np.float32)
            self._fft_n = next_fast_len(M + self.L - 1)
            self._H = np.fft.rfft(self.h, self._fft_n)
        out = np.empty((M, C), dtype=np.float32)
        for c in range(C):
            block = np.concatenate([self._tail[:, c], chunk[:, c]])
            Y = np.fft.rfft(block, self._fft_n) * self._H
            y = np.fft.irfft(Y, self._fft_n)
            out[:, c] = y[self.L - 1: self.L - 1 + M]
        self._tail = (chunk[-(self.L - 1):] if M >= self.L - 1
                      else np.concatenate([self._tail[M:], chunk])).copy()
        return out


class StreamingDelay:
    def __init__(self, D: int):
        self.D = D
        self._buf = None

    def process(self, chunk: np.ndarray) -> np.ndarray:
        M, C = chunk.shape
        if self._buf is None:
            self._buf = np.zeros((self.D, C), dtype=np.float32)
        combined = np.concatenate([self._buf, chunk])
        out = combined[:M].copy()
        self._buf = combined[M: M + self.D].copy()
        return out


# ---------------------------------------------------------------------------
# STFT air-patch: recover tamed high-frequency content from algo into apollo
# ---------------------------------------------------------------------------
class STFTAirPatcher:
    """
    In the air band (above crossover_high), patches algo content into apollo
    wherever apollo's energy has been tamed below threshold_db.
    """
    def __init__(self, nperseg: int = 1024, overlap: int = 8,
                 threshold_db: float = STFT_THRESHOLD_DB):
        self.nperseg = nperseg
        self.noverlap = int(nperseg * (1 - 1 / overlap))
        self.threshold_db = threshold_db
        self._tail_len = self.noverlap
        self._apollo_tail = None
        self._algo_tail = None

    def process(self, apollo: np.ndarray, algo: np.ndarray) -> np.ndarray:
        M, C = apollo.shape
        if self._apollo_tail is None:
            self._apollo_tail = np.zeros((self._tail_len, C), dtype=np.float32)
            self._algo_tail   = np.zeros((self._tail_len, C), dtype=np.float32)

        pad = self.nperseg
        a_ext  = np.concatenate([self._apollo_tail, apollo, np.zeros((pad, C), np.float32)])
        al_ext = np.concatenate([self._algo_tail,   algo,   np.zeros((pad, C), np.float32)])

        out = np.empty((M, C), dtype=np.float32)
        for c in range(C):
            _, _, Za  = scipy_stft(a_ext[:, c],  fs=SAMPLE_RATE, nperseg=self.nperseg,
                                   noverlap=self.noverlap, scaling='psd')
            _, _, Zal = scipy_stft(al_ext[:, c], fs=SAMPLE_RATE, nperseg=self.nperseg,
                                   noverlap=self.noverlap, scaling='psd')
            # where apollo is tamed, substitute algo
            mask = 20.0 * np.log10(np.maximum(np.abs(Za), 1e-20)) <= self.threshold_db
            Zo = np.where(mask, Zal, Za)
            _, y = scipy_istft(Zo, fs=SAMPLE_RATE, nperseg=self.nperseg,
                               noverlap=self.noverlap, scaling='psd')
            out[:, c] = y[self._tail_len: self._tail_len + M]

        self._apollo_tail = apollo[-self._tail_len:].copy()
        self._algo_tail   = algo[-self._tail_len:].copy()
        return out


# ---------------------------------------------------------------------------
# M/S
# ---------------------------------------------------------------------------
def ms_encode(s: np.ndarray) -> np.ndarray:
    return np.column_stack(((s[:, 0] + s[:, 1]) * 0.5,
                             (s[:, 0] - s[:, 1]) * 0.5))

def ms_decode(ms: np.ndarray) -> np.ndarray:
    return np.column_stack((ms[:, 0] + ms[:, 1],
                             ms[:, 0] - ms[:, 1]))


# ---------------------------------------------------------------------------
# Transient detection (per-chunk, per-channel)
# ---------------------------------------------------------------------------
def transient_weight(chunk: np.ndarray, threshold: float = TRANSIENT_THRESHOLD) -> float:
    """Returns a scalar in [0, 1] indicating transient strength."""
    rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
    peak = np.max(np.abs(chunk))
    ratio = peak / (rms + 1e-12)
    # ratio > ~3 = transient
    strength = np.clip((ratio / (1.0 / threshold) - 1.0), 0.0, 1.0)
    return float(strength)


# ---------------------------------------------------------------------------
# DeltaWave analysis — find crossover from difference spectrum
# ---------------------------------------------------------------------------
def analyze_deltawave(delta_path: Path, fs: int) -> tuple[float, float]:
    """
    Reads a DeltaWave difference file and finds the frequency band where
    the two sources diverge most (highest difference energy).
    Returns (crossover_low_hz, crossover_high_hz).
    """
    print(f"[deltawave] analyzing {delta_path.name}...", file=sys.stderr)
    data, dfs = sf.read(str(delta_path), always_2d=True)
    if dfs != fs:
        print(f"[deltawave] sample rate mismatch ({dfs} vs {fs}), results may be approximate",
              file=sys.stderr)

    mono = data.mean(axis=1)
    n = len(mono)
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(mono * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / dfs)

    # Smooth spectrum in log-frequency bins
    n_bins = 200
    log_freqs = np.logspace(np.log10(20), np.log10(dfs / 2), n_bins)
    bin_energy = np.zeros(n_bins)
    for i in range(n_bins - 1):
        mask = (freqs >= log_freqs[i]) & (freqs < log_freqs[i + 1])
        if mask.any():
            bin_energy[i] = np.mean(spectrum[mask] ** 2)

    # Find peak divergence region
    peak_bin = int(np.argmax(bin_energy))
    # Crossover low: where energy starts rising (10% of peak), high: where it falls back
    peak_e = bin_energy[peak_bin]
    low_bin = peak_bin
    for i in range(peak_bin, 0, -1):
        if bin_energy[i] < peak_e * 0.1:
            low_bin = i
            break
    high_bin = peak_bin
    for i in range(peak_bin, n_bins - 1):
        if bin_energy[i] < peak_e * 0.1:
            high_bin = i
            break

    low_hz  = float(np.clip(log_freqs[low_bin],  2000, 12000))
    high_hz = float(np.clip(log_freqs[high_bin], low_hz + 1000, 20000))
    print(f"[deltawave] detected crossover: {low_hz:.0f} Hz – {high_hz:.0f} Hz", file=sys.stderr)
    return low_hz, high_hz


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def align_sources(apollo: np.ndarray, algo: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Xcorr-based sample-accurate alignment. Returns (apollo, algo) trimmed to same length."""
    MAX_OFFSET = fs // 4  # 0.25s search window
    mono_a = apollo[:, 0].astype(np.float64)
    mono_b = algo[:, 0].astype(np.float64)
    n = min(len(mono_a), len(mono_b), fs * 10)  # use first 10s for detection
    ref, test = mono_b[:n] - mono_b[:n].mean(), mono_a[:n] - mono_a[:n].mean()
    corr = fftconvolve(ref, test[::-1], mode='full')
    mid = len(test) - 1
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
# ---------------------------------------------------------------------------
class BlendState:
    def __init__(self, crossover_low: float, crossover_high: float):
        self.cl = crossover_low
        self.ch = crossover_high
        D1 = (max(2049, 513 * 4 | 1) - 1) // 2  # group delay of FIRs

        # Low-pass: apollo source for lows
        self.lp_apollo = StreamingFIR(_design_fir("lp", crossover_low, SAMPLE_RATE))
        self.lp_algo   = StreamingFIR(_design_fir("lp", crossover_low, SAMPLE_RATE))

        # Band-pass middle: blend zone (lp at high, hp at low)
        self.bp_lp_apollo = StreamingFIR(_design_fir("lp", crossover_high, SAMPLE_RATE))
        self.bp_lp_algo   = StreamingFIR(_design_fir("lp", crossover_high, SAMPLE_RATE))
        self.bp_hp_apollo = StreamingFIR(_design_fir("hp", crossover_low,  SAMPLE_RATE))
        self.bp_hp_algo   = StreamingFIR(_design_fir("hp", crossover_low,  SAMPLE_RATE))

        # High-pass: algo source for air
        self.hp_apollo = StreamingFIR(_design_fir("hp", crossover_high, SAMPLE_RATE))
        self.hp_algo   = StreamingFIR(_design_fir("hp", crossover_high, SAMPLE_RATE))

        # STFT air patcher (patches algo air into apollo where apollo is tamed)
        self.air_patcher = STFTAirPatcher()

        # Delay compensation so all bands are time-aligned at reconstruction
        self.delay_D1 = D1
        self.delay_apollo = StreamingDelay(D1)
        self.delay_algo   = StreamingDelay(D1)

        self.total_delay = D1


def process_chunk(apollo_ms: np.ndarray, algo_ms: np.ndarray,
                  st: BlendState, sample_pos: int) -> np.ndarray:
    M = apollo_ms.shape[0]

    # Delay raw signals to match FIR group delay
    apollo_d = st.delay_apollo.process(apollo_ms)
    algo_d   = st.delay_algo.process(algo_ms)

    # Low band — Apollo (cleaner, fewer artifacts)
    low_apollo = st.lp_apollo.process(apollo_ms)
    # complement from algo for low — not used directly, reconstructed via perfect sum
    # High band — algo (preserves air/ambience)
    high_algo = st.hp_algo.process(algo_ms)

    # Air patch: recover tamed apollo high-end from algo
    high_patched = st.air_patcher.process(
        st.hp_apollo.process(apollo_ms),
        high_algo
    )

    # Mid band from both (hp at low crossover, lp at high crossover)
    mid_apollo = st.bp_hp_apollo.process(st.bp_lp_apollo.process(apollo_ms))
    mid_algo   = st.bp_hp_algo.process(st.bp_lp_algo.process(algo_ms))

    # Transient detection: boost algo weight in mid-band during transients
    t_strength = transient_weight(apollo_d)
    algo_mid_weight = np.float32(0.35 + t_strength * TRANSIENT_ALGO_BOOST)
    apollo_mid_weight = np.float32(1.0 - algo_mid_weight)

    mid_blend = mid_apollo * apollo_mid_weight + mid_algo * algo_mid_weight

    # Reconstruct: low (apollo) + mid (blend) + high (patched algo air)
    out = low_apollo + mid_blend + high_patched

    return out


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------
def load_audio(path: Path, target_sr: int) -> np.ndarray:
    data, sr = sf.read(str(path), always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    if data.shape[1] > 2:
        data = data[:, :2]
    if sr != target_sr:
        print(f"[load] {path.name}: resampling {sr} -> {target_sr}", file=sys.stderr)
        import subprocess, tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, data, sr, subtype='FLOAT')
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as out:
            out_path = out.name
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error',
                        '-i', tmp_path, '-ar', str(target_sr), '-y', out_path], check=True)
        data, _ = sf.read(out_path, always_2d=True)
        os.unlink(tmp_path)
        os.unlink(out_path)
    return data.astype(np.float32)


def write_output(data: np.ndarray, output_path: Path, format_name: str,
                 bitrate: str | None = None, bit_depth: int | None = None) -> None:
    fmt = FORMAT_DEFAULTS[format_name]
    if format_name == 'wav':
        sf.write(str(output_path), data, SAMPLE_RATE, subtype='FLOAT')
        return
    # encode via ffmpeg
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, data, SAMPLE_RATE, subtype='FLOAT')
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', tmp_path, '-c:a', fmt['codec']]
    br = bitrate or fmt.get('default_bitrate')
    bd = bit_depth or fmt.get('default_bit_depth')
    if br:
        cmd.extend(['-b:a', br])
    if bd and format_name == 'flac':
        cmd.extend(['-sample_fmt', f's{bd}'])
    cmd.extend(['-y', str(output_path)])
    subprocess.run(cmd, check=True)
    import os; os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Blend Apollo + algorithmic restorations")
    parser.add_argument('--apollo',     required=True, help='Apollo-restored WAV')
    parser.add_argument('--algo',       required=True, help='Algorithmic restoration WAV')
    parser.add_argument('--output',     required=True, help='Output file path')
    parser.add_argument('--deltawave',  default=None,  help='DeltaWave difference WAV for crossover calibration')
    parser.add_argument('--crossover',  default=None,  type=float,
                        help='Manual crossover Hz (overrides auto-detection, sets single point)')
    parser.add_argument('--format',     default='wav', choices=list(FORMAT_DEFAULTS),
                        help='Output format (default: wav)')
    parser.add_argument('--bitrate',    default=None,  help='Bitrate for lossy output')
    parser.add_argument('--bit-depth',  default=None,  type=int, help='Bit depth for flac')
    parser.add_argument('--chunk-size', default=CHUNK_SIZE, type=int)
    args = parser.parse_args()

    apollo_path = Path(args.apollo)
    algo_path   = Path(args.algo)
    output_path = Path(args.output)

    # Determine crossovers
    if args.crossover is not None:
        cl = max(1000.0, args.crossover - 3000.0)
        ch = args.crossover
        print(f"[blend] manual crossover: {cl:.0f}–{ch:.0f} Hz", file=sys.stderr)
    elif args.deltawave is not None:
        cl, ch = analyze_deltawave(Path(args.deltawave), SAMPLE_RATE)
    else:
        cl, ch = DEFAULT_CROSSOVER_LOW, DEFAULT_CROSSOVER_HIGH
        print(f"[blend] using default crossovers: {cl:.0f}–{ch:.0f} Hz", file=sys.stderr)

    # Load
    print(f"[blend] loading {apollo_path.name}...", file=sys.stderr)
    apollo = load_audio(apollo_path, SAMPLE_RATE)
    print(f"[blend] loading {algo_path.name}...", file=sys.stderr)
    algo   = load_audio(algo_path,   SAMPLE_RATE)

    # Align
    print(f"[blend] aligning sources...", file=sys.stderr)
    apollo, algo = align_sources(apollo, algo, SAMPLE_RATE)
    total_samples = len(apollo)
    print(f"[blend] {total_samples / SAMPLE_RATE:.1f}s to process", file=sys.stderr)

    # Init pipeline
    st = BlendState(cl, ch)
    output_chunks = []
    skip = st.total_delay
    sample_pos = 0
    chunk_size = args.chunk_size

    # Process
    for i in range(0, total_samples, chunk_size):
        a_chunk = apollo[i: i + chunk_size]
        b_chunk = algo[i: i + chunk_size]
        n = min(len(a_chunk), len(b_chunk))
        a_chunk, b_chunk = a_chunk[:n], b_chunk[:n]

        # Pad last chunk to 1024 boundary for STFT
        rem = n % 1024
        if rem:
            pad = 1024 - rem
            a_chunk = np.pad(a_chunk, ((0, pad), (0, 0)))
            b_chunk = np.pad(b_chunk, ((0, pad), (0, 0)))

        a_ms = ms_encode(a_chunk)
        b_ms = ms_encode(b_chunk)
        out_ms = process_chunk(a_ms, b_ms, st, sample_pos)
        out = ms_decode(out_ms)

        if skip > 0:
            if out.shape[0] <= skip:
                skip -= out.shape[0]
                continue
            out = out[skip:]
            skip = 0

        output_chunks.append(out[:n] if rem else out)
        sample_pos += n

        pct = min(100, i * 100 // total_samples)
        print(f"\r  {pct:3d}% — {i / SAMPLE_RATE:.1f}s / {total_samples / SAMPLE_RATE:.1f}s",
              end='', file=sys.stderr)

    # Flush delay
    flush_n = ((st.total_delay + 1023) // 1024) * 1024
    zeros = np.zeros((flush_n, CHANNELS), dtype=np.float32)
    flush_ms = process_chunk(zeros, zeros, st, sample_pos)
    flush = ms_decode(flush_ms)
    output_chunks.append(flush[:st.total_delay])

    print(f"\r  done — {total_samples / SAMPLE_RATE:.1f}s processed" + " " * 20, file=sys.stderr)

    result = np.concatenate(output_chunks, axis=0)
    result = np.clip(result, -1.0, 1.0)

    # Write
    print(f"[blend] writing {output_path}...", file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_output(result, output_path, args.format,
                 bitrate=args.bitrate, bit_depth=args.bit_depth)
    print(f"[blend] done → {output_path}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
