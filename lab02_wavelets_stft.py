"""Lab 02 (skeleton): Wavelets (Haar) + STFT bridge."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal
from scipy.signal import stft
import cv2
import numpy as np
import numpy.typing as npt

ThresholdMode = Literal["soft", "hard"]


def haar_dwt1(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-level 1D Haar DWT.

    For odd-length inputs, pad one sample (edge/reflect policy, document choice).

    Args:
        x: 1D numeric signal.

    Returns:
        (approx, detail): each length ~N/2.
    """
    signal_arr = np.asarray(x, dtype=np.float64)
    if signal_arr.ndim != 1:
        raise ValueError("Input must be 1D")
    if signal_arr.shape[0] % 2 == 1:
        signal_arr = np.concatenate([signal_arr, signal_arr[-1:]])
    c = np.sqrt(0.5)
    even = signal_arr[::2]
    odd = signal_arr[1::2]
    return c * (even + odd), c * (even - odd)


def haar_idwt1(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """
    Invert one-level 1D Haar DWT.

    Args:
        approx: Approximation coefficients.
        detail: Detail coefficients.

    Returns:
        Reconstructed signal.
    """
    a = np.asarray(approx, dtype=np.float64)
    d = np.asarray(detail, dtype=np.float64)
    if len(a) != len(d):
        raise ValueError("approx and detail lengths differ")
    c = np.sqrt(0.5)
    reconstructed = np.empty(2 * len(a), dtype=np.float64)
    reconstructed[::2] = c * (a + d)
    reconstructed[1::2] = c * (a - d)
    return reconstructed


def haar_dwt2(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute one-level 2D separable Haar DWT for grayscale images.

    Args:
        image: 2D grayscale image.

    Returns:
        LL, (LH, HL, HH).
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image must be 2D")
    if img.shape[0] % 2:
        img = np.pad(img, ((0, 1), (0, 0)), mode="edge")
    if img.shape[1] % 2:
        img = np.pad(img, ((0, 0), (0, 1)), mode="edge")
    a = img[0::2, 0::2]
    b = img[0::2, 1::2]
    c = img[1::2, 0::2]
    d = img[1::2, 1::2]
    LL = (a + b + c + d) / 2.0
    LH = (a + b - c - d) / 2.0
    HL = (a - b + c - d) / 2.0
    HH = (a - b - c + d) / 2.0
    return LL, (LH, HL, HH)


def haar_idwt2(LL: np.ndarray, bands: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Invert one-level 2D Haar DWT.

    Args:
        LL: Low-low sub-band.
        bands: Tuple `(LH, HL, HH)`.

    Returns:
        Reconstructed image (crop policy for odd sizes should be documented).
    """
    LH, HL, HH = bands
    ll = np.asarray(LL, dtype=np.float64)
    lh = np.asarray(LH, dtype=np.float64)
    hl = np.asarray(HL, dtype=np.float64)
    hh = np.asarray(HH, dtype=np.float64)
    out = np.empty((ll.shape[0] * 2, ll.shape[1] * 2), dtype=np.float64)
    out[0::2, 0::2] = (ll + lh + hl + hh) / 2.0
    out[0::2, 1::2] = (ll + lh - hl - hh) / 2.0
    out[1::2, 0::2] = (ll - lh + hl - hh) / 2.0
    out[1::2, 1::2] = (ll - lh - hl + hh) / 2.0
    return out


def wavelet_threshold(coeffs: Any, threshold: float, mode: ThresholdMode = "soft") -> Any:
    """
    Apply thresholding to coefficient arrays.

    Args:
        coeffs: Array or nested tuples/lists of arrays.
        threshold: Non-negative threshold value.
        mode: `"soft"` or `"hard"`.

    Returns:
        Thresholded coefficients with same structure.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if isinstance(coeffs, np.ndarray):
        mag = np.abs(coeffs)
        if mode == "hard":
            filtered = coeffs * (mag >= threshold)
        elif mode == "soft":
            filtered = np.sign(coeffs) * np.maximum(mag - threshold, 0.0)
        else:
            raise ValueError("mode must be 'soft' or 'hard'")
        return filtered.astype(coeffs.dtype, copy=False)
    if isinstance(coeffs, list):
        return [wavelet_threshold(item, threshold, mode) for item in coeffs]
    if isinstance(coeffs, tuple):
        return tuple(wavelet_threshold(item, threshold, mode) for item in coeffs)
    raise TypeError(f"Unsupported type: {type(coeffs)}")


def wavelet_denoise(image: np.ndarray, levels: int, threshold: float, mode: ThresholdMode = "soft") -> np.ndarray:
    """
    Denoise image via multi-level Haar thresholding.

    Args:
        image: 2D grayscale image.
        levels: Number of decomposition levels.
        threshold: Coefficient threshold.
        mode: `"soft"` or `"hard"`.

    Returns:
        Denoised image with deterministic behavior.
    """
    shape = image.shape
    dtype = image.dtype
    low = np.asarray(image, dtype=np.float64)
    pyramid: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    low_shapes: list[tuple[int, int]] = []
    for _level in range(levels):
        low_shapes.append(low.shape)
        low, detail = haar_dwt2(low)
        pyramid.append(detail)
    for detail, target_shape in zip(pyramid[::-1], low_shapes[::-1], strict=False):
        clean_detail = wavelet_threshold(detail, threshold, mode)
        low = haar_idwt2(low, clean_detail)
        low = low[: target_shape[0], : target_shape[1]]
    low = low[: shape[0], : shape[1]]
    return low.astype(dtype, copy=False)


def stft1(
    x: np.ndarray,
    fs_hz: float,
    frame_len: int,
    hop_len: int,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT for 1D signal using SciPy.

    Returns:
        `(freqs_hz, times_s, Zxx)` where `Zxx` is complex.
    """
    noverlap = frame_len - hop_len
    if noverlap < 0:
        raise ValueError("hop_len cannot be larger than frame_len")
    return stft(
        np.asarray(x),
        fs=fs_hz,
        window=window,
        nperseg=frame_len,
        noverlap=noverlap,
    )


def spectrogram_magnitude(Zxx: np.ndarray, log_scale: bool = True) -> np.ndarray:
    """
    Convert STFT matrix to magnitude spectrogram.

    Args:
        Zxx: Complex STFT matrix.
        log_scale: If True, return `log(1 + magnitude)`.

    Returns:
        Non-negative finite magnitude matrix.
    """
    spec = np.abs(np.asarray(Zxx))
    if log_scale:
        spec = np.log1p(spec)
    return spec


def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Min-max normalize an array to `[0,255]` for visualization."""
    values = np.asarray(x, dtype=np.float64)
    mn = float(values.min())
    mx = float(values.max())
    if mx <= mn:
        return np.zeros_like(values, dtype=np.uint8)
    return np.round((values - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)


def main() -> int:
    """
    Lab 02 demo (skeleton).

    Expected behavior after implementation:
    - wavelet denoising demo on image from `./imgs/`
    - LL/LH/HL/HH band visualization
    - STFT spectrogram demo on synthetic chirp signal
    - save outputs to `./out/lab02/` (no GUI windows)
    """
    parser = argparse.ArgumentParser(description="Lab 02 skeleton (implement functions first).")
    parser.add_argument("--img", type=str, default="lenna.png", help="Image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab02", help="Output directory (relative to repo root)")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(imgs_dir / args.img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(imgs_dir / args.img))

    missing: list[str] = []

    # --- Wavelet demo ---
    try:
        rng = np.random.default_rng(0)
        noisy = img.astype(np.float32) + rng.normal(0.0, 20.0, size=img.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 255.0)
        den = wavelet_denoise(noisy, levels=2, threshold=20.0, mode="soft")

        ll, (lh, hl, hh) = haar_dwt2(img.astype(np.float32))

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Original", img),
                ("Noisy (Gaussian)", noisy),
                ("Wavelet denoised", den),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(im), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "wavelet_denoise.png")

        plt.figure(figsize=(10, 8))
        for i, (title, band) in enumerate(
            [
                ("LL", ll),
                ("LH", lh),
                ("HL", hl),
                ("HH", hh),
            ],
            start=1,
        ):
            plt.subplot(2, 2, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(band), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "wavelet_bands.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    # --- STFT bridge demo ---
    try:
        fs = 400.0
        duration_s = 2.0
        t = np.arange(int(fs * duration_s), dtype=np.float64) / fs
        f0, f1 = 15.0, 120.0
        k = (f1 - f0) / duration_s
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
        x = np.sin(phase)

        freqs, times, zxx = stft1(x, fs_hz=fs, frame_len=128, hop_len=32, window="hann")
        mag = spectrogram_magnitude(zxx, log_scale=True)

        plt.figure(figsize=(8, 4))
        plt.pcolormesh(times, freqs, mag, shading="gouraud")
        plt.title("STFT Spectrogram (log-magnitude)")
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")
        plt.colorbar(label="log(1 + |Zxx|)")
        save_figure(out_dir / "stft_spectrogram.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 02 demo is incomplete. Implement the TODO functions in labs/lab02_wavelets_stft.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
