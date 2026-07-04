import argparse
import os
from typing import Dict, Tuple

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya bulunamadı: {path}")

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Görüntü okunamadı veya desteklenmeyen format.")
    return image


def detect_fluorescent_regions(image: np.ndarray, threshold: int = 180, min_area: int = 50) -> Dict[str, float]:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    region_areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) >= min_area]
    bright_mask = binary == 255
    mean_intensity = float(np.mean(image[bright_mask])) if np.any(bright_mask) else 0.0

    return {
        "count": len(region_areas),
        "total_area": int(sum(region_areas)),
        "mean_intensity": mean_intensity,
    }


def has_signal(image_path: str, threshold: int = 180, min_area: int = 50) -> Tuple[bool, Dict[str, float]]:
    image = load_image(image_path)
    summary = detect_fluorescent_regions(image, threshold, min_area)
    return summary["count"] > 0, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Western blot jel görüntüsünde etidyum bromür sinyalini tespit eder.")
    parser.add_argument("--image", required=True, help="Analiz edilecek jel görüntüsü")
    parser.add_argument("--threshold", type=int, default=180, help="Parlaklık eşiği (varsayılan 180)")
    parser.add_argument("--min-area", type=int, default=50, help="Bölge için minimum alan (varsayılan 50)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal, summary = has_signal(args.image, args.threshold, args.min_area)

    print("--- Western Blot Sinyal Tespiti ---")
    print(f"Görüntü: {args.image}")
    print(f"Eşik: {args.threshold}")
    print(f"Minimum alan: {args.min_area}")
    print(f"Bölge sayısı: {summary['count']}")
    print(f"Toplam alan: {summary['total_area']}")
    print(f"Ortalama parlaklık: {summary['mean_intensity']:.2f}")

    if signal:
        print("Sinyal tespit edildi: Etidyum bromür ile ışımaya sahip protein bantları bulundu.")
        exit(0)
    print("Sinyal bulunamadı.")
    exit(1)


if __name__ == "__main__":
    main()
