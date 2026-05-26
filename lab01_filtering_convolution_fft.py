from __future__ import annotations

"""Lab 01 (skeleton): filtering/convolution + FFT tools (spatial & frequency domain)."""

import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import numpy.typing as npt
from scipy import signal

BorderType = Literal["reflect", "constant", "wrap", "replicate"]


def conv2d(image: npt.NDArray[np.generic], kernel: npt.NDArray[np.generic], border: BorderType = "reflect") -> np.ndarray:
    """
    2D convolution for grayscale/color images (spatial-domain linear filtering).

    Args:
        image: `(H,W)` or `(H,W,C)` array (any numeric dtype; computed in `float32`).
        kernel: `(kH,kW)` 2D kernel (any numeric dtype).
        border: `"reflect" | "constant" | "wrap" | "replicate"`.

    Returns:
        `float32` array with the same shape as `image`.
    """
    img = np.asarray(image, dtype=np.float32)
    k = np.asarray(kernel, dtype=np.float32)
    if img.ndim not in (2, 3):
        raise ValueError("image must be 2D or 3D")
    if k.ndim != 2:
        raise ValueError("kernel must be 2D")

    top = k.shape[0] // 2
    bottom = k.shape[0] - top - 1
    left = k.shape[1] // 2
    right = k.shape[1] - left - 1
    modes = {"reflect": "reflect", "constant": "constant", "wrap": "wrap", "replicate": "edge"}
    if border not in modes:
        raise ValueError(f"bad border mode: {border}")

    if img.ndim == 2:
        pad_width = ((top, bottom), (left, right))
    else:
        pad_width = ((top, bottom), (left, right), (0, 0))
    if border == "constant":
        padded = np.pad(img, pad_width, mode="constant")
    else:
        padded = np.pad(img, pad_width, mode=modes[border])

    cv_kernel = np.flip(k, axis=(0, 1))
    tmp = cv2.filter2D(padded, ddepth=cv2.CV_32F, kernel=cv_kernel, borderType=cv2.BORDER_CONSTANT)
    return tmp[top: top + img.shape[0], left: left + img.shape[1]].astype(np.float32)


def make_gaussian_kernel(ksize: int, sigma: float) -> npt.NDArray[np.float32]:
    """
    Create a normalized 2D Gaussian kernel (sum ~ 1).

    Args:
        ksize: Positive odd kernel size.
        sigma: Standard deviation in pixels (> 0).

    Returns:
        `(ksize, ksize)` `float32` kernel.
    """
    if ksize < 1 or ksize % 2 != 1:
        raise ValueError("ksize must be positive and odd")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    g = cv2.getGaussianKernel(ksize, sigma, ktype=cv2.CV_32F)
    return (g @ g.T).astype(np.float32)


def _clip_to_dtype_range(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    target = np.dtype(dtype)
    values = np.asarray(x)
    if np.issubdtype(target, np.integer):
        info = np.iinfo(target)
        values = np.minimum(np.maximum(values, info.min), info.max)
    return values.astype(target, copy=False)


def apply_gaussian_blur(image: npt.NDArray[np.generic], ksize: int, sigma: float) -> np.ndarray:
    """
    Gaussian smoothing in the spatial domain (via `conv2d`).

    Args:
        image: `(H,W)` or `(H,W,C)` image.
        ksize: Positive odd kernel size.
        sigma: Standard deviation in pixels.

    Returns:
        Same shape/dtype as input.
    """
    original = np.asarray(image)
    smoothed = conv2d(original, make_gaussian_kernel(ksize, sigma), border="reflect")
    return _clip_to_dtype_range(np.round(smoothed), original.dtype)


def apply_box_blur(image: npt.NDArray[np.generic], ksize: int) -> np.ndarray:
    """
    Box/mean blur using a `(ksize x ksize)` uniform kernel (via `conv2d`).

    Args:
        image: `(H,W)` or `(H,W,C)` image.
        ksize: Positive odd window size.

    Returns:
        Same shape/dtype as input.
    """
    if ksize < 1 or ksize % 2 != 1:
        raise ValueError("ksize must be positive and odd")
    original = np.asarray(image)
    weights = np.ones((ksize, ksize), dtype=np.float32) / np.float32(ksize * ksize)
    smoothed = conv2d(original, weights, border="reflect")
    return _clip_to_dtype_range(np.round(smoothed), original.dtype)


def apply_median_blur(image: npt.NDArray[np.generic], ksize: int) -> np.ndarray:
    """
    Median filter (best for salt-and-pepper noise).

    Args:
        image: `uint8` image (grayscale or color).
        ksize: Positive odd neighborhood size.

    Returns:
        Same shape/dtype as input.
    """
    if ksize < 1 or ksize % 2 != 1:
        raise ValueError("ksize must be positive and odd")
    img = np.asarray(image)
    return cv2.medianBlur(img, ksize)


def add_salt_pepper_noise(
    image: npt.NDArray[np.generic],
    amount: float,
    salt_vs_pepper: float = 0.5,
    *,
    seed: int = 0,
) -> np.ndarray:
    """
    Add salt-and-pepper (impulse) noise (deterministic by `seed`).

    Args:
        image: Input image (any numeric dtype).
        amount: Fraction of pixels to corrupt in `[0, 1]`.
        salt_vs_pepper: Probability of "salt" among corrupted pixels.
        seed: RNG seed.

    Returns:
        Noised image with the same shape/dtype.
    """
    if amount < 0 or amount > 1:
        raise ValueError("amount must be between 0 and 1")
    if salt_vs_pepper < 0 or salt_vs_pepper > 1:
        raise ValueError("salt_vs_pepper must be between 0 and 1")
    src = np.asarray(image)
    result = src.copy()
    rng = np.random.default_rng(seed)
    h, w = src.shape[:2]
    n_bad = int(amount * h * w)
    if n_bad <= 0:
        return result

    coords = rng.choice(h * w, n_bad, replace=False)
    ys, xs = np.unravel_index(coords, (h, w))
    split = int(n_bad * salt_vs_pepper)

    if np.issubdtype(src.dtype, np.integer):
        dtype_info = np.iinfo(src.dtype)
        lo, hi = dtype_info.min, dtype_info.max
    else:
        lo = 0.0
        hi = 1.0 if float(np.max(src)) <= 1.0 else 255.0

    result[ys[:split], xs[:split]] = hi
    result[ys[split:], xs[split:]] = lo
    return result


def add_gaussian_noise(image: npt.NDArray[np.generic], sigma: float, *, seed: int = 0) -> np.ndarray:
    """
    Add zero-mean Gaussian noise (deterministic by `seed`).

    Args:
        image: Input image (any numeric dtype).
        sigma: Standard deviation (>= 0), in image intensity units.
        seed: RNG seed.

    Returns:
        Noised image with the same shape/dtype.
    """
    if sigma < 0:
        raise ValueError("sigma must not be negative")
    base = np.asarray(image)
    noise = np.random.default_rng(seed).normal(loc=0.0, scale=sigma, size=base.shape)
    return _clip_to_dtype_range(np.round(base.astype(np.float32) + noise), base.dtype)


def sobel_edges(image: npt.NDArray[np.generic], ksize: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sobel gradients and magnitude (edge strength).

    Returns:
        `(gx, gy, magnitude)` as `float32` arrays of shape `(H, W)`.

    Args:
        image: Input image (converted to grayscale internally).
        ksize: Positive odd Sobel size.
    """
    if ksize < 1 or ksize % 2 != 1:
        raise ValueError("ksize must be positive and odd")
    src = np.asarray(image)
    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src = src.astype(np.float32, copy=False)
    gx = cv2.Sobel(src, cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gy = cv2.Sobel(src, cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    return gx, gy, np.sqrt(gx * gx + gy * gy).astype(np.float32)


def laplacian_edges(image: npt.NDArray[np.generic], ksize: int = 3) -> np.ndarray:
    """
    Laplacian edge response (absolute value).

    Args:
        image: Input image (converted to grayscale internally).
        ksize: Positive odd aperture size.

    Returns:
        `float32` array `(H, W)` (non-negative).
    """
    if ksize < 1 or ksize % 2 != 1:
        raise ValueError("ksize must be positive and odd")
    src = np.asarray(image)
    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    response = cv2.Laplacian(src.astype(np.float32), cv2.CV_32F, ksize=ksize)
    return np.abs(response).astype(np.float32)


def fft2_image(image: npt.NDArray[np.generic]) -> np.ndarray:
    """
    Compute the 2D DFT using OpenCV, returning a 2-channel float32 spectrum.

    Returns:
        spectrum: (H, W, 2) float32 array where spectrum[...,0] is Re and spectrum[...,1] is Im.

    Args:
        image: Input image (grayscale or color). Converted to grayscale internally.
    """
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(arr.astype(np.float32))
    return np.dstack((fft.real, fft.imag)).astype(np.float32)


def fftshift2(spectrum: npt.NDArray[np.floating]) -> np.ndarray:
    """
    Shift the zero-frequency component to the center.

    Args:
        spectrum: A 2D array `(H,W)` or a 3D array `(H,W,2)` (OpenCV DFT format).

    Returns:
        Spectrum with quadrants swapped so that DC is at the center.
    """
    shifted = np.fft.fftshift(np.asarray(spectrum), axes=(0, 1))
    return shifted.astype(np.asarray(spectrum).dtype, copy=False)


def magnitude_spectrum(spectrum: npt.NDArray[np.floating], log_scale: bool = True) -> np.ndarray:
    """
    Convert a 2-channel OpenCV DFT spectrum into a magnitude image.

    Args:
        spectrum: OpenCV DFT output in shape `(H, W, 2)` with Re/Im channels.
        log_scale: If True, returns `log(1 + magnitude)` which is the standard way to
            visualize FFT spectra with large dynamic range.

    Returns:
        `float32` array of shape `(H, W)` with non-negative values.
    """
    s = np.asarray(spectrum)
    if s.ndim != 3 or s.shape[-1] != 2:
        raise ValueError("OpenCV-style spectrum must be (H,W,2)")
    mag = np.hypot(s[..., 0], s[..., 1])
    return np.log1p(mag).astype(np.float32) if log_scale else mag.astype(np.float32)


def ideal_low_pass_filter(shape: tuple[int, int] | tuple[int, int, int], cutoff_radius: float) -> np.ndarray:
    """
    Create an ideal (hard) low-pass frequency-domain mask.

    Args:
        shape: Target shape, typically the DFT spectrum shape `(H,W,2)` or `(H,W)`.
        cutoff_radius: Cutoff radius in pixels (in the frequency plane).

    Returns:
        A `float32` mask of shape `(H, W, 2)` suitable for elementwise multiplication
        with an OpenCV DFT spectrum.
    """
    if cutoff_radius < 0:
        raise ValueError("cutoff_radius must be >= 0")
    height, width = shape[:2]
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    dist2 = (y - height // 2) ** 2 + (x - width // 2) ** 2
    plane = (dist2 <= cutoff_radius * cutoff_radius).astype(np.float32)
    return np.dstack((plane, plane))


def ideal_high_pass_filter(shape: tuple[int, int] | tuple[int, int, int], cutoff_radius: float) -> np.ndarray:
    """
    Create an ideal (hard) high-pass frequency-domain mask.

    This is defined as `1 - ideal_low_pass_filter(...)`.
    """
    low = ideal_low_pass_filter(shape, cutoff_radius)
    high = np.ones_like(low, dtype=np.float32)
    high -= low
    return high


def apply_frequency_filter(image: npt.NDArray[np.generic], filter_mask: npt.NDArray[np.floating]) -> np.ndarray:
    """
    Filter an image in the frequency domain using an (H,W) or (H,W,2) mask.

    Args:
        image: Input image (grayscale or color). Converted to grayscale internally.
        filter_mask:
            Either:
            - `(H, W)` single-channel mask (will be broadcast to 2 channels), or
            - `(H, W, 2)` OpenCV-compatible 2-channel mask.

    Returns:
        Filtered spatial-domain image as `float32` of shape `(H, W)`.
    """
    src = np.asarray(image)
    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    spectrum = np.fft.fftshift(np.fft.fft2(src.astype(np.float32)))
    mask = np.asarray(filter_mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != src.shape[:2]:
        raise ValueError("filter_mask has incompatible shape")
    restored = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    return restored.astype(np.float32)


def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """
    Min-max normalize an array to `[0, 255]` (`uint8`) for visualization.

    Args:
        x: Any numeric array-like.

    Returns:
        2D/3D array (same shape as input) scaled to `uint8`.
    """
    arr = np.asarray(x, dtype=np.float32)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val == min_val:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - min_val) / (max_val - min_val)
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def main() -> int:
    """
    Lab 01 demo (skeleton).

    Expected behavior after implementation:
    - load 1-2 images from `./imgs/`
    - synthesize salt&pepper + Gaussian noise
    - compare median vs Gaussian vs box blur
    - compute Sobel/Laplacian edges
    - visualize FFT magnitude spectrum and apply ideal LPF/HPF
    - save all outputs into `./out/lab01/` (no GUI windows)
    """
    parser = argparse.ArgumentParser(description="Lab 01 skeleton (implement functions first).")
    parser.add_argument("--img1", type=str, default="lenna.png", help="First image from ./imgs/")
    parser.add_argument("--img2", type=str, default="airplane.bmp", help="Second image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab01", help="Output directory (relative to repo root)")
    args = parser.parse_args()

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    def normalize_to_uint8(x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
        arr = np.asarray(x, dtype=np.float32)
        mn, mx = float(np.min(arr)), float(np.max(arr))
        if mx <= mn:
            return np.zeros_like(arr, dtype=np.uint8)
        y = (arr - mn) * (255.0 / (mx - mn))
        return np.clip(y, 0.0, 255.0).astype(np.uint8)

    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img1 = cv2.imread(str(imgs_dir / args.img1), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(imgs_dir / args.img2), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        raise FileNotFoundError(str(imgs_dir / args.img1))
    if img2 is None:
        raise FileNotFoundError(str(imgs_dir / args.img2))

    missing: list[str] = []

    # --- Noise + denoise comparisons ---
    try:
        sp_noisy = add_salt_pepper_noise(img1, amount=0.08, salt_vs_pepper=0.55, seed=0)
        g_noisy = add_gaussian_noise(img1, sigma=15.0, seed=0)

        median_sp = apply_median_blur(sp_noisy, 5)
        gauss_sp = apply_gaussian_blur(sp_noisy, 5, 1.2)
        box_sp = apply_box_blur(sp_noisy, 5)

        gauss_g = apply_gaussian_blur(g_noisy, 5, 1.2)
        box_g = apply_box_blur(g_noisy, 5)

        plt.figure(figsize=(12, 6))
        for i, (title, im) in enumerate(
            [
                ("Original", img1),
                ("Salt & pepper", sp_noisy),
                ("Median (5x5)", median_sp),
                ("Gaussian (5,σ=1.2)", gauss_sp),
                ("Box (5x5)", box_sp),
                ("Gaussian noise", g_noisy),
            ],
            start=1,
        ):
            plt.subplot(2, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "denoise_sp_and_examples.png")

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Gaussian noise", g_noisy),
                ("Gaussian blur", gauss_g),
                ("Box blur", box_g),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "denoise_gaussian_noise.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    # --- Edge detection ---
    try:
        gx, gy, mag = sobel_edges(img2, ksize=3)
        _ = (gx, gy)
        lap = laplacian_edges(img2, ksize=3)

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Input", img2),
                ("Sobel magnitude", normalize_to_uint8(mag)),
                ("Laplacian |·|", normalize_to_uint8(lap)),
            ],
            start=1,
        ):
            plt.subplot(1, 3, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "edges.png")

        cv2.imwrite(str(out_dir / "sobel_mag.png"), normalize_to_uint8(mag))
        cv2.imwrite(str(out_dir / "laplacian_abs.png"), normalize_to_uint8(lap))
    except NotImplementedError as exc:
        missing.append(str(exc))

    # --- FFT + frequency-domain filtering ---
    try:
        spec = fft2_image(img2)
        spec_shift = fftshift2(spec)
        mag = magnitude_spectrum(spec_shift, log_scale=True)

        lp = ideal_low_pass_filter(spec_shift.shape, cutoff_radius=30.0)
        hp = ideal_high_pass_filter(spec_shift.shape, cutoff_radius=30.0)
        lowpassed = apply_frequency_filter(img2, lp)
        highpassed = apply_frequency_filter(img2, hp)

        plt.figure(figsize=(12, 4))
        for i, (title, im) in enumerate(
            [
                ("Input", img2),
                ("Magnitude spectrum (log)", normalize_to_uint8(mag)),
                ("LPF result", normalize_to_uint8(lowpassed)),
                ("HPF result", normalize_to_uint8(highpassed)),
            ],
            start=1,
        ):
            plt.subplot(1, 4, i)
            plt.title(title)
            plt.imshow(im, cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "fft_frequency_filters.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 01 demo is incomplete. Implement the TODO functions in labs/lab01_filtering_convolution_fft.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
