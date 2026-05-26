from __future__ import annotations

"""Lab 05 (skeleton): motion estimation with dense optical flow."""

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def optical_flow_farneback(prev_gray: np.ndarray, next_gray: np.ndarray, **params: Any) -> np.ndarray:
    """
    Compute dense optical flow using Farneback algorithm.

    Flow convention:
    - output[..., 0] = horizontal displacement `dx`
    - output[..., 1] = vertical displacement `dy`

    Args:
        prev_gray: Previous frame (grayscale image).
        next_gray: Next frame (grayscale image).
        **params: Optional Farneback parameter overrides.

    Returns:
        Dense flow field `(H, W, 2)` as float array.
    """
    cfg: dict[str, Any] = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    cfg.update(params)
    prev = np.asarray(prev_gray)
    nxt = np.asarray(next_gray)
    return cv2.calcOpticalFlowFarneback(
        prev,
        nxt,
        None,
        float(cfg["pyr_scale"]),
        int(cfg["levels"]),
        int(cfg["winsize"]),
        int(cfg["iterations"]),
        int(cfg["poly_n"]),
        float(cfg["poly_sigma"]),
        int(cfg["flags"]),
    )


def flow_to_hsv(flow_xy: np.ndarray) -> np.ndarray:
    """
    Convert flow field to BGR visualization via HSV mapping.

    Args:
        flow_xy: Dense flow `(H,W,2)`.

    Returns:
        `uint8` BGR image `(H,W,3)` suitable for `cv2.imwrite`.
    """
    flow = np.asarray(flow_xy, dtype=np.float32)
    dx = flow[..., 0]
    dy = flow[..., 1]
    magnitude = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)
    angle = np.mod(angle, 2.0 * np.pi)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = (angle * 90.0 / np.pi).astype(np.uint8)
    hsv[:, :, 1] = 255
    max_magnitude = float(magnitude.max())
    if max_magnitude > 0.0:
        hsv[:, :, 2] = np.clip(magnitude / max_magnitude * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main() -> int:
    """
    Lab 05 demo (skeleton).

    Expected behavior after implementation:
    - load image from `./imgs/` as previous frame
    - create next frame with known translation
    - compute Farneback optical flow
    - save prev/next/flow visualization to `./out/lab05/`
    """
    parser = argparse.ArgumentParser(description="Lab 05 skeleton (implement functions first).")
    parser.add_argument("--img", type=str, default="airplane.bmp", help="Input image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab05", help="Output directory (relative to repo root)")
    parser.add_argument("--dx", type=float, default=5.0, help="Horizontal translation (pixels)")
    parser.add_argument("--dy", type=float, default=3.0, help="Vertical translation (pixels)")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(imgs_dir / args.img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(imgs_dir / args.img))

    missing: list[str] = []

    try:
        prev = img
        h, w = prev.shape
        M = np.array([[1.0, 0.0, float(args.dx)], [0.0, 1.0, float(args.dy)]], dtype=np.float32)
        nxt = cv2.warpAffine(prev, M, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        flow = optical_flow_farneback(prev, nxt)
        vis = flow_to_hsv(flow)

        cv2.imwrite(str(out_dir / "prev.png"), prev)
        cv2.imwrite(str(out_dir / "next.png"), nxt)
        cv2.imwrite(str(out_dir / "flow_vis.png"), vis)
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 05 demo is incomplete. Implement the TODO functions in labs/lab05_motion_estimation.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
