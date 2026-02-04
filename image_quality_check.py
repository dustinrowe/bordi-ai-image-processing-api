import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def readability_prefilter(
    img_bgr: np.ndarray,
    min_mean: float = 100.0,
    max_blur: float = 20.0,
    max_noise: float = 1.0,
) -> Tuple[bool, Dict[str, float]]:
    # Hard-coded crop to remove fixed white borders.
    # These values are tuned for the Manage Asset Drilldown images (2500x2900).
    x1, x2 = 500, 2056
    h, w = img_bgr.shape[:2]
    x1_clamped = max(0, min(x1, w))
    x2_clamped = max(0, min(x2, w))
    if x2_clamped > x1_clamped:
        img_bgr = img_bgr[:, x1_clamped:x2_clamped]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mean = float(gray.mean())
    std = float(gray.std())
    blur = blur_score(gray)
    blue_mean = float(img_bgr[:, :, 0].mean())
    dark_pixel_ratio = float((gray < 15).mean())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    high_freq = cv2.subtract(gray, cv2.GaussianBlur(gray, (5, 5), 0))
    noise_estimate = float(high_freq.std())

    metrics = {
        "mean": mean,
        "std": std,
        "blur": blur,
        "blue_mean": blue_mean,
        "dark_pixel_ratio": dark_pixel_ratio,
        "edge_density": edge_density,
        "noise_estimate": noise_estimate,
    }

    quality_ok = mean > min_mean and blur < max_blur and noise_estimate < max_noise
    return quality_ok, metrics


def check_images(
    image_paths: List[str],
    min_mean: float = 100.0,
    max_blur: float = 20.0,
    max_noise: float = 1.0,
) -> None:
    results = []
    for path in image_paths:
        if not os.path.exists(path):
            results.append({"path": path, "ok": False, "error": "missing"})
            continue

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            results.append({"path": path, "ok": False, "error": "unreadable"})
            continue

        ok, metrics = readability_prefilter(
            img_bgr, min_mean=min_mean, max_blur=max_blur, max_noise=max_noise
        )
        results.append(
            {
                "path": path,
                "ok": ok,
                "mean": round(metrics["mean"], 4),
                "std": round(metrics["std"], 4),
                "blur": round(metrics["blur"], 4),
                "blue_mean": round(metrics["blue_mean"], 4),
                "dark_pixel_ratio": round(metrics["dark_pixel_ratio"], 6),
                "edge_density": round(metrics["edge_density"], 6),
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
