from __future__ import annotations

"""Lab 04 (skeleton): Markov Random Field (MRF) image restoration."""

import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

PenaltyType = Literal["quadratic", "huber"]


def mrf_energy(
    x: np.ndarray,
    y: np.ndarray,
    lambda_smooth: float,
    penalty: PenaltyType = "quadratic",
    huber_delta: float = 1.0,
) -> float:
    """
    Compute pairwise MRF energy for grayscale image restoration.

    Energy:
        E(x) = sum_p (x_p - y_p)^2 + lambda * sum_(p,q in N) rho(x_p - x_q)

    Args:
        x: Restored image candidate `(H,W)`.
        y: Observed noisy image `(H,W)`.
        lambda_smooth: Smoothness weight.
        penalty: `"quadratic"` or `"huber"`.
        huber_delta: Delta parameter for Huber penalty.

    Returns:
        Scalar energy.
    """
    estimate = np.asarray(x, dtype=np.float64)
    noisy = np.asarray(y, dtype=np.float64)
    if estimate.shape != noisy.shape:
        raise ValueError("x and y shapes are different")
    energy = float(np.sum((estimate - noisy) ** 2))

    def rho(values: np.ndarray) -> np.ndarray:
        if penalty == "quadratic":
            return values ** 2
        if penalty == "huber":
            a = np.abs(values)
            quadratic = 0.5 * values ** 2
            linear = huber_delta * a - 0.5 * huber_delta ** 2
            return np.where(a <= huber_delta, quadratic, linear)
        raise ValueError("unknown penalty")

    energy += float(lambda_smooth * np.sum(rho(estimate[:, 1:] - estimate[:, :-1])))
    energy += float(lambda_smooth * np.sum(rho(estimate[1:, :] - estimate[:-1, :])))
    return energy


def mrf_denoise(
    y: np.ndarray,
    lambda_smooth: float,
    num_iters: int,
    step: float = 0.1,
    penalty: PenaltyType = "quadratic",
    huber_delta: float = 1.0,
) -> np.ndarray:
    """
    Restore grayscale image by minimizing MRF energy.

    Args:
        y: Observed noisy image `(H,W)`.
        lambda_smooth: Smoothness weight.
        num_iters: Number of optimization iterations.
        step: Optimization step size.
        penalty: `"quadratic"` or `"huber"`.
        huber_delta: Delta parameter for Huber penalty.

    Returns:
        Restored image with the same shape as `y`.
    """
    noisy = np.asarray(y, dtype=np.float64)
    estimate = noisy.copy()

    def rho_grad(values: np.ndarray) -> np.ndarray:
        if penalty == "quadratic":
            return 2.0 * values
        if penalty == "huber":
            return np.clip(values, -huber_delta, huber_delta)
        raise ValueError("unknown penalty")

    for _iteration in range(num_iters):
        grad = 2.0 * (estimate - noisy)

        left_to_right = estimate[:, 1:] - estimate[:, :-1]
        g = rho_grad(left_to_right)
        grad[:, 1:] += lambda_smooth * g
        grad[:, :-1] -= lambda_smooth * g

        top_to_bottom = estimate[1:, :] - estimate[:-1, :]
        g = rho_grad(top_to_bottom)
        grad[1:, :] += lambda_smooth * g
        grad[:-1, :] -= lambda_smooth * g

        estimate = estimate - step * grad
    return estimate.astype(y.dtype, copy=False)


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0,255] uint8 for visualization."""
    values = np.asarray(x, dtype=np.float64)
    mn, mx = float(np.min(values)), float(np.max(values))
    if mx == mn:
        return np.zeros_like(values, dtype=np.uint8)
    normalized = (values - mn) / (mx - mn)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def main() -> int:
    """
    Lab 04 demo (skeleton).

    Expected behavior after implementation:
    - load grayscale image from `./imgs/`
    - add Gaussian noise (deterministic seed)
    - denoise with MRF (quadratic and/or huber)
    - save side-by-side result to `./out/lab04/mrf_denoise.png`
    """
    parser = argparse.ArgumentParser(description="Lab 04 skeleton (implement functions first).")
    parser.add_argument("--img", type=str, default="lenna.png", help="Input image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab04", help="Output directory (relative to repo root)")
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

    try:
        clean = img.astype(np.float32)
        rng = np.random.default_rng(0)
        noisy = clean + rng.normal(0.0, 18.0, size=clean.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 255.0)

        den_quad = mrf_denoise(noisy, lambda_smooth=0.25, num_iters=80, step=0.1, penalty="quadratic")
        den_hub = mrf_denoise(noisy, lambda_smooth=0.25, num_iters=80, step=0.1, penalty="huber", huber_delta=8.0)

        e_noisy_q = mrf_energy(noisy, noisy, lambda_smooth=0.25, penalty="quadratic")
        e_quad = mrf_energy(den_quad, noisy, lambda_smooth=0.25, penalty="quadratic")
        e_noisy_h = mrf_energy(noisy, noisy, lambda_smooth=0.25, penalty="huber", huber_delta=8.0)
        e_hub = mrf_energy(den_hub, noisy, lambda_smooth=0.25, penalty="huber", huber_delta=8.0)

        plt.figure(figsize=(12, 4))
        panels = [
            ("Original", clean),
            ("Noisy (seed=0)", noisy),
            (f"MRF quadratic\nE: {e_noisy_q:.1f} -> {e_quad:.1f}", den_quad),
            (f"MRF huber\nE: {e_noisy_h:.1f} -> {e_hub:.1f}", den_hub),
        ]
        for i, (title, im) in enumerate(panels, start=1):
            plt.subplot(1, 4, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(im), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "mrf_denoise.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 04 demo is incomplete. Implement the TODO functions in labs/lab04_mrf_restoration.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
