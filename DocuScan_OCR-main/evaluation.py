from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import easyocr
import numpy as np
import pytesseract

DEFAULT_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

EASY_READER_CACHE: dict[tuple[str, ...], easyocr.Reader] = {}


# -----------------------------
# OCR setup
# -----------------------------
def configure_tesseract(tesseract_cmd: Optional[str] = None) -> str:
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return tesseract_cmd

    for path in DEFAULT_TESSERACT_PATHS:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return path

    return getattr(pytesseract.pytesseract, "tesseract_cmd", "")


def get_easy_reader(langs: Optional[list[str]] = None) -> easyocr.Reader:
    if langs is None:
        langs = ["en"]

    key = tuple(sorted({lang.strip() for lang in langs if lang.strip()}))
    if not key:
        key = ("en",)

    if key not in EASY_READER_CACHE:
        EASY_READER_CACHE[key] = easyocr.Reader(list(key), gpu=False, verbose=False)

    return EASY_READER_CACHE[key]


# -----------------------------
# Text metrics
# -----------------------------
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().split())



def levenshtein_distance(s1: str, s2: str) -> int:
    s1 = s1 or ""
    s2 = s2 or ""

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]



def character_error_rate(gt: str, pred: str) -> float:
    gt = normalize_text(gt)
    pred = normalize_text(pred)

    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    return levenshtein_distance(gt, pred) / len(gt)



def word_level_accuracy(gt: str, pred: str) -> float:
    gt_words = normalize_text(gt).split()
    pred_words = normalize_text(pred).split()

    if len(gt_words) == 0:
        return 1.0 if len(pred_words) == 0 else 0.0

    correct = 0
    for i in range(min(len(gt_words), len(pred_words))):
        if gt_words[i] == pred_words[i]:
            correct += 1

    return correct / len(gt_words)


# -----------------------------
# Document pipeline
# -----------------------------
def reorder_points(points: np.ndarray) -> np.ndarray:
    points = np.array(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]

    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]

    return rect



def four_point_warp(image_bgr: np.ndarray, points: np.ndarray) -> np.ndarray:
    rect = reorder_points(points)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b), 1)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b), 1)

    target = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, target)
    return cv2.warpPerspective(image_bgr, matrix, (max_width, max_height))



def draw_document_box(image_bgr: np.ndarray, points: np.ndarray, found_document: bool) -> np.ndarray:
    output = image_bgr.copy()
    color = (0, 255, 0) if found_document else (0, 165, 255)
    points_int = points.astype(int)
    cv2.polylines(output, [points_int], True, color, 4)
    for x, y in points_int:
        cv2.circle(output, (x, y), 8, (255, 0, 0), -1)
    return output



def detect_and_warp_document(
    image_bgr: np.ndarray,
    blur_kernel: int = 5,
    canny_low: int = 75,
    canny_high: int = 200,
    max_contours: int = 10,
) -> dict[str, Any]:
    blur_kernel = max(3, int(blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blurred, int(canny_low), int(canny_high))

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[: max(1, int(max_contours))]

    document_quad = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            document_quad = approx.reshape(4, 2)
            break

    if document_quad is None:
        h, w = image_bgr.shape[:2]
        document_quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        found_document = False
        detection_status = "Fallback to full image"
        warped = image_bgr.copy()
    else:
        found_document = True
        detection_status = "Largest quadrilateral detected"
        warped = four_point_warp(image_bgr, document_quad)

    document_box = draw_document_box(image_bgr, document_quad, found_document)

    return {
        "gray": gray,
        "blurred": blurred,
        "edges": edges,
        "warped": warped,
        "document_box": document_box,
        "points": reorder_points(document_quad),
        "found_document": found_document,
        "detection_status": detection_status,
    }



def enhance_for_ocr(
    warped_bgr: np.ndarray,
    denoise_strength: int = 20,
    block_size: int = 11,
    c_value: int = 2,
) -> np.ndarray:
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, max(1, int(denoise_strength)), 7, 21)

    block_size = max(3, int(block_size))
    if block_size % 2 == 0:
        block_size += 1

    return cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        int(c_value),
    )


# -----------------------------
# OCR
# -----------------------------
def run_tesseract(
    image_gray: np.ndarray,
    language: str = "eng",
    config: str = "--psm 6",
) -> tuple[str, float]:
    start = time.time()
    text = pytesseract.image_to_string(image_gray, lang=language, config=config)
    return normalize_text(text), time.time() - start



def run_easyocr(
    image_gray: np.ndarray,
    languages: Optional[list[str]] = None,
    paragraph: bool = True,
) -> tuple[str, float]:
    reader = get_easy_reader(languages or ["en"])
    start = time.time()
    results = reader.readtext(image_gray, detail=0, paragraph=paragraph)
    return normalize_text(" ".join(results)), time.time() - start


# -----------------------------
# Shared comparison logic
# -----------------------------
def compare_ocr_on_image(
    image_bgr: np.ndarray,
    ground_truth: str = "",
    tesseract_cmd: Optional[str] = None,
    tesseract_lang: str = "eng",
    tesseract_config: str = "--psm 6",
    easyocr_langs: Optional[list[str]] = None,
    blur_kernel: int = 5,
    canny_low: int = 75,
    canny_high: int = 200,
    denoise_strength: int = 20,
    threshold_block_size: int = 11,
    threshold_c: int = 2,
) -> dict[str, Any]:
    tess_path = configure_tesseract(tesseract_cmd)
    pipeline = detect_and_warp_document(
        image_bgr,
        blur_kernel=blur_kernel,
        canny_low=canny_low,
        canny_high=canny_high,
    )
    enhanced = enhance_for_ocr(
        pipeline["warped"],
        denoise_strength=denoise_strength,
        block_size=threshold_block_size,
        c_value=threshold_c,
    )

    gt_text = normalize_text(ground_truth)
    tess_text, tess_time = run_tesseract(enhanced, language=tesseract_lang, config=tesseract_config)
    easy_text, easy_time = run_easyocr(enhanced, languages=easyocr_langs or ["en"])

    result = {
        "ground_truth": gt_text,
        "tesseract_path": tess_path,
        "detection_status": pipeline["detection_status"],
        "found_document": pipeline["found_document"],
        "gray": pipeline["gray"],
        "blurred": pipeline["blurred"],
        "edges": pipeline["edges"],
        "document_box": pipeline["document_box"],
        "warped": pipeline["warped"],
        "enhanced": enhanced,
        "tesseract_text": tess_text,
        "tesseract_time_sec": tess_time,
        "dl_text": easy_text,
        "dl_time_sec": easy_time,
    }

    if gt_text:
        result.update(
            {
                "tesseract_word_acc": word_level_accuracy(gt_text, tess_text),
                "tesseract_cer": character_error_rate(gt_text, tess_text),
                "dl_word_acc": word_level_accuracy(gt_text, easy_text),
                "dl_cer": character_error_rate(gt_text, easy_text),
            }
        )

    return result



def save_single_report(
    result: dict[str, Any],
    tesseract_lang: str,
    easyocr_langs: list[str],
    output_path: Optional[str] = None,
) -> str:
    lines = [
        "OCR Report",
        "=" * 60,
        f"Document detection: {result['detection_status']}",
        f"Tesseract language: {tesseract_lang}",
        f"EasyOCR languages: {', '.join(easyocr_langs)}",
        "",
    ]

    if result.get("ground_truth"):
        lines.extend(["Ground truth:", result["ground_truth"], ""])

    lines.extend([
        "Tesseract text:",
        result["tesseract_text"],
        f"Time: {result['tesseract_time_sec']:.4f} s",
    ])
    if result.get("ground_truth"):
        lines.extend([
            f"Word Accuracy: {result['tesseract_word_acc']:.4f}",
            f"CER: {result['tesseract_cer']:.4f}",
        ])
    lines.append("")

    lines.extend([
        "EasyOCR text:",
        result["dl_text"],
        f"Time: {result['dl_time_sec']:.4f} s",
    ])
    if result.get("ground_truth"):
        lines.extend([
            f"Word Accuracy: {result['dl_word_acc']:.4f}",
            f"CER: {result['dl_cer']:.4f}",
        ])

    if output_path is None:
        fd, output_path = tempfile.mkstemp(prefix="ocr_report_", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write("\n".join(lines))
    else:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines))

    return output_path


# -----------------------------
# Dataset evaluation
# -----------------------------
def evaluate_single_image(
    image_path: str | Path,
    gt_text_path: str | Path,
    tesseract_cmd: Optional[str] = None,
    tesseract_lang: str = "eng",
    tesseract_config: str = "--psm 6",
    easyocr_langs: Optional[list[str]] = None,
    blur_kernel: int = 5,
    canny_low: int = 75,
    canny_high: int = 200,
    denoise_strength: int = 20,
    threshold_block_size: int = 11,
    threshold_c: int = 2,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    with open(gt_text_path, "r", encoding="utf-8") as file:
        ground_truth = file.read()

    result = compare_ocr_on_image(
        image,
        ground_truth=ground_truth,
        tesseract_cmd=tesseract_cmd,
        tesseract_lang=tesseract_lang,
        tesseract_config=tesseract_config,
        easyocr_langs=easyocr_langs or ["en"],
        blur_kernel=blur_kernel,
        canny_low=canny_low,
        canny_high=canny_high,
        denoise_strength=denoise_strength,
        threshold_block_size=threshold_block_size,
        threshold_c=threshold_c,
    )
    result["image_name"] = Path(image_path).name
    return result



def evaluate_dataset(
    images_dir: str | Path,
    gt_dir: str | Path,
    output_report: str = "ocr_report_refactored.txt",
    tesseract_cmd: Optional[str] = None,
    tesseract_lang: str = "eng",
    tesseract_config: str = "--psm 6",
    easyocr_langs: Optional[list[str]] = None,
    blur_kernel: int = 5,
    canny_low: int = 75,
    canny_high: int = 200,
    denoise_strength: int = 20,
    threshold_block_size: int = 11,
    threshold_c: int = 2,
) -> None:
    images_dir = Path(images_dir)
    gt_dir = Path(gt_dir)

    print("images_dir =", images_dir.resolve())
    print("gt_dir     =", gt_dir.resolve())

    image_files = sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
        ]
    )

    if not image_files:
        print("No test images found.")
        return

    results: list[dict[str, Any]] = []
    for image_path in image_files:
        gt_path = gt_dir / f"{image_path.stem}.txt"
        if not gt_path.exists():
            print(f"GT file missing for: {image_path.name}")
            continue

        try:
            result = evaluate_single_image(
                image_path,
                gt_path,
                tesseract_cmd=tesseract_cmd,
                tesseract_lang=tesseract_lang,
                tesseract_config=tesseract_config,
                easyocr_langs=easyocr_langs or ["en"],
                blur_kernel=blur_kernel,
                canny_low=canny_low,
                canny_high=canny_high,
                denoise_strength=denoise_strength,
                threshold_block_size=threshold_block_size,
                threshold_c=threshold_c,
            )
            results.append(result)
            print(f"Done: {image_path.name}")
            print(f"Document detection: {result['detection_status']}")
            print(f"Tesseract Word Acc: {result['tesseract_word_acc']:.4f}")
            print(f"Tesseract CER     : {result['tesseract_cer']:.4f}")
            print(f"EasyOCR Word Acc  : {result['dl_word_acc']:.4f}")
            print(f"EasyOCR CER       : {result['dl_cer']:.4f}")
        except Exception as exc:
            print(f"Error on {image_path.name}: {exc}")

    if not results:
        print("No valid results.")
        return

    avg_tess_word = sum(item["tesseract_word_acc"] for item in results) / len(results)
    avg_tess_cer = sum(item["tesseract_cer"] for item in results) / len(results)
    avg_tess_time = sum(item["tesseract_time_sec"] for item in results) / len(results)

    avg_easy_word = sum(item["dl_word_acc"] for item in results) / len(results)
    avg_easy_cer = sum(item["dl_cer"] for item in results) / len(results)
    avg_easy_time = sum(item["dl_time_sec"] for item in results) / len(results)

    with open(output_report, "w", encoding="utf-8") as file:
        file.write("OCR Report\n")
        file.write("=" * 60 + "\n\n")

        for item in results:
            file.write(f"Image: {item['image_name']}\n")
            file.write(f"Document detection: {item['detection_status']}\n")
            file.write(f"Ground Truth: {item['ground_truth']}\n\n")

            file.write("[Tesseract]\n")
            file.write(f"Predicted Text: {item['tesseract_text']}\n")
            file.write(f"Word Accuracy: {item['tesseract_word_acc']:.4f}\n")
            file.write(f"CER: {item['tesseract_cer']:.4f}\n")
            file.write(f"Time (sec): {item['tesseract_time_sec']:.4f}\n\n")

            file.write("[EasyOCR]\n")
            file.write(f"Predicted Text: {item['dl_text']}\n")
            file.write(f"Word Accuracy: {item['dl_word_acc']:.4f}\n")
            file.write(f"CER: {item['dl_cer']:.4f}\n")
            file.write(f"Time (sec): {item['dl_time_sec']:.4f}\n")
            file.write("-" * 60 + "\n\n")

        file.write("FINAL AVERAGES\n")
        file.write("=" * 60 + "\n")
        file.write(f"Tesseract Avg Word Accuracy: {avg_tess_word:.4f}\n")
        file.write(f"Tesseract Avg CER: {avg_tess_cer:.4f}\n")
        file.write(f"Tesseract Avg Time: {avg_tess_time:.4f}\n\n")
        file.write(f"EasyOCR Avg Word Accuracy: {avg_easy_word:.4f}\n")
        file.write(f"EasyOCR Avg CER: {avg_easy_cer:.4f}\n")
        file.write(f"EasyOCR Avg Time: {avg_easy_time:.4f}\n")

    print("Report saved to:", Path(output_report).resolve())


if __name__ == "__main__":
    print("Evaluation started...")
    evaluate_dataset(
        images_dir="test_images",
        gt_dir="ground_truth",
        output_report="ocr_report_refactored.txt",
    )
