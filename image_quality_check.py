import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel = kernel.astype(np.float32)
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    shape = (image.shape[0], image.shape[1], kh, kw)
    strides = (
        padded.strides[0],
        padded.strides[1],
        padded.strides[0],
        padded.strides[1],
    )
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.einsum("ijkl,kl->ij", windows, kernel)


def blur_score(gray: np.ndarray) -> float:
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lap = convolve2d(gray, laplacian_kernel)
    return float(lap.var())


def readability_prefilter(
    img_rgb: np.ndarray,
    min_mean: float = 100.0,
    max_blur: float = 25.0,
    max_noise: float = 1.5,
) -> Tuple[bool, Dict[str, float]]:
    # Hard-coded crop to remove fixed white borders.
    # These values are tuned for the Manage Asset Drilldown images (2500x2900).
    x1, x2 = 500, 2056
    h, w = img_rgb.shape[:2]
    x1_clamped = max(0, min(x1, w))
    x2_clamped = max(0, min(x2, w))
    if x2_clamped > x1_clamped:
        img_rgb = img_rgb[:, x1_clamped:x2_clamped]

    gray = (
        0.299 * img_rgb[:, :, 0]
        + 0.587 * img_rgb[:, :, 1]
        + 0.114 * img_rgb[:, :, 2]
    ).astype(np.float32)

    mean = float(gray.mean())
    std = float(gray.std())
    blur = blur_score(gray)
    gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
    blur_gray = np.array(Image.fromarray(gray_uint8).filter(ImageFilter.GaussianBlur(radius=1)))
    high_freq = gray - blur_gray.astype(np.float32)
    noise_estimate = float(high_freq.std())

    metrics = {
        "mean": mean,
        "std": std,
        "blur": blur,
        "noise_estimate": noise_estimate,
    }

    quality_ok = mean > min_mean and blur < max_blur and noise_estimate < max_noise
    return quality_ok, metrics


def check_images(
    image_paths: List[str],
    min_mean: float = 100.0,
    max_blur: float = 25.0,
    max_noise: float = 1.5,
) -> None:
    results = []
    for path in image_paths:
        if not os.path.exists(path):
            results.append({"path": path, "ok": False, "error": "missing"})
            continue

        try:
            img_rgb = np.array(Image.open(path).convert("RGB"))
        except Exception:
            results.append({"path": path, "ok": False, "error": "unreadable"})
            continue

        ok, metrics = readability_prefilter(
            img_rgb, min_mean=min_mean, max_blur=max_blur, max_noise=max_noise
        )
        results.append(
            {
                "path": path,
                "ok": ok,
                "mean": round(metrics["mean"], 4),
                "std": round(metrics["std"], 4),
                "blur": round(metrics["blur"], 4),
                "noise_estimate": round(metrics["noise_estimate"], 4),
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    default_images = [
        "/Users/dflow/Downloads/Manage Asset Drilldown (4).jpeg",
        "/Users/dflow/Downloads/Manage Asset Drilldown (3).jpeg",
        "/Users/dflow/Downloads/Manage Asset Drilldown (1).jpeg",
        '/Users/dflow/Downloads/Advanced Search Asset 006.jpg',
        '/Users/dflow/Downloads/Manage Asset Drilldown (2).jpeg'
    ]
    check_images(default_images)
