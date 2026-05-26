from __future__ import annotations

"""Lab 03 (skeleton): geometric transforms + ORB features/matching + homography."""

import argparse
from pathlib import Path

import cv2
import numpy as np


def warp_affine(image: np.ndarray, M: np.ndarray, out_shape: tuple[int, int], border: str = "reflect") -> np.ndarray:
    """
    Warp image with affine transform.

    Args:
        image: Grayscale or color image.
        M: Affine matrix `(2,3)`.
        out_shape: Output shape `(out_h, out_w)`.
        border: Border mode: reflect/constant/replicate.

    Returns:
        Warped image.
    """
    if border == "constant":
        border_mode = cv2.BORDER_CONSTANT
    elif border == "replicate":
        border_mode = cv2.BORDER_REPLICATE
    else:
        border_mode = cv2.BORDER_REFLECT_101
    height, width = out_shape
    matrix = np.asarray(M, dtype=np.float64)
    return cv2.warpAffine(image, matrix, dsize=(width, height), borderMode=border_mode)


def warp_perspective(image: np.ndarray, H: np.ndarray, out_shape: tuple[int, int], border: str = "reflect") -> np.ndarray:
    """
    Warp image with perspective homography.

    Args:
        image: Grayscale or color image.
        H: Homography matrix `(3,3)`.
        out_shape: Output shape `(out_h, out_w)`.
        border: Border mode: reflect/constant/replicate.

    Returns:
        Warped image.
    """
    if border == "constant":
        border_mode = cv2.BORDER_CONSTANT
    elif border == "replicate":
        border_mode = cv2.BORDER_REPLICATE
    else:
        border_mode = cv2.BORDER_REFLECT_101
    height, width = out_shape
    matrix = np.asarray(H, dtype=np.float64)
    return cv2.warpPerspective(image, matrix, dsize=(width, height), borderMode=border_mode)


def detect_orb(image: np.ndarray, n_features: int = 500) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """
    Detect ORB keypoints and descriptors.

    Args:
        image: Grayscale or BGR image.
        n_features: Max number of ORB keypoints.

    Returns:
        `(keypoints, descriptors)`, where descriptors may be `None`.
    """
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    orb = cv2.ORB_create(nfeatures=max(1, int(n_features)))
    return orb.detectAndCompute(gray, None)


def match_descriptors(
    desc1: np.ndarray | None,
    desc2: np.ndarray | None,
    method: str = "bf_hamming",
    ratio_test: float = 0.75,
) -> list[cv2.DMatch]:
    """
    Match descriptors using BFMatcher + ratio test.

    Args:
        desc1: Query descriptors.
        desc2: Train descriptors.
        method: Matching method (`bf_hamming`).
        ratio_test: Lowe ratio threshold.

    Returns:
        Good matches sorted by distance.
    """
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(desc1, desc2, k=2)
    good_matches: list[cv2.DMatch] = []
    for item in raw:
        if len(item) != 2:
            continue
        first, other = item
        if first.distance / max(other.distance, 1e-12) < ratio_test:
            good_matches.append(first)
    return sorted(good_matches, key=lambda m: (m.distance, m.queryIdx, m.trainIdx))


def estimate_homography_from_matches(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    ransac_thresh: float = 3.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Estimate homography from matches with RANSAC.

    Args:
        kp1: Keypoints in source image.
        kp2: Keypoints in destination image.
        matches: Matches from source to destination.
        ransac_thresh: Reprojection threshold in pixels.

    Returns:
        `(H, inlier_mask)` or `(None, None)`.
    """
    if len(matches) < 4:
        return None, None
    points1 = []
    points2 = []
    for match in matches:
        points1.append(kp1[match.queryIdx].pt)
        points2.append(kp2[match.trainIdx].pt)
    src_pts = np.float32(points1).reshape(-1, 1, 2)
    dst_pts = np.float32(points2).reshape(-1, 1, 2)
    H_mat, inlier_mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    return H_mat, None if inlier_mask is None else inlier_mask.ravel()


def main() -> int:
    """
    Lab 03 demo (skeleton).

    Expected behavior after implementation:
    - affine transform demo (rotate+translate)
    - perspective warp demo (homography)
    - ORB detect + matching + homography estimation visualization
    - save outputs to `./out/lab03/` (no GUI windows)
    """
    parser = argparse.ArgumentParser(description="Lab 03 skeleton (implement functions first).")
    parser.add_argument("--img", type=str, default="lenna.png", help="Input image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab03", help="Output directory (relative to repo root)")
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

    img_bgr = cv2.imread(str(imgs_dir / args.img), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(str(imgs_dir / args.img))

    h, w = img_bgr.shape[:2]
    missing: list[str] = []

    # --- Geometric warps ---
    try:
        center = (w / 2.0, h / 2.0)
        m = cv2.getRotationMatrix2D(center, angle=15.0, scale=0.95)
        m[0, 2] += 18.0
        m[1, 2] += 10.0
        aff = warp_affine(img_bgr, m, out_shape=(h, w), border="reflect")
        cv2.imwrite(str(out_dir / "affine_warp.png"), aff)

        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([[12, 18], [w - 30, 8], [w - 18, h - 24], [20, h - 10]])
        hmat = cv2.getPerspectiveTransform(src, dst)
        per = warp_perspective(img_bgr, hmat, out_shape=(h, w), border="reflect")
        cv2.imwrite(str(out_dir / "perspective_warp.png"), per)
    except NotImplementedError as exc:
        missing.append(str(exc))

    # --- ORB + matching + homography ---
    try:
        kp1, d1 = detect_orb(img_bgr, n_features=1000)
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([[12, 18], [w - 30, 8], [w - 18, h - 24], [20, h - 10]])
        hmat = cv2.getPerspectiveTransform(src, dst)
        warped = warp_perspective(img_bgr, hmat, out_shape=(h, w), border="reflect")

        kp2, d2 = detect_orb(warped, n_features=1000)
        matches = match_descriptors(d1, d2, method="bf_hamming", ratio_test=0.75)
        h_est, inliers = estimate_homography_from_matches(kp1, kp2, matches, ransac_thresh=3.0)

        if inliers is not None:
            draw_matches = [m for m, keep in zip(matches, inliers, strict=False) if int(keep) > 0]
        else:
            draw_matches = matches
        draw_matches = draw_matches[:80]

        vis = cv2.drawMatches(
            img_bgr,
            kp1,
            warped,
            kp2,
            draw_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        plt.figure(figsize=(12, 6))
        plt.title(
            f"ORB matches (good={len(matches)}, inliers={int(np.sum(inliers)) if inliers is not None else 0}, "
            f"H={'ok' if h_est is not None else 'None'})"
        )
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        save_figure(out_dir / "orb_matches_homography.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 03 demo is incomplete. Implement the TODO functions in labs/lab03_geometry_features_matching.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
