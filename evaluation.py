import time
from pathlib import Path

import cv2
import pytesseract
import easyocr


# =========================
# =========================
_easy_reader = None

def get_easy_reader(langs=(["ar","fa","ur","ug","en"])):
    global _easy_reader
    if _easy_reader is None:
        print("Loading EasyOCR model...")
        _easy_reader = easyocr.Reader(list(langs), gpu=False)
    return _easy_reader


# =========================
# =========================
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    text = " ".join(text.split())
    return text


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
        return 0.0 if len(pred_words) > 0 else 1.0

    correct = 0
    total = len(gt_words)

    for i in range(min(len(gt_words), len(pred_words))):
        if gt_words[i] == pred_words[i]:
            correct += 1

    return correct / total


# =========================
# OCR methods
# =========================
def run_tesseract(image):
    start = time.time()
    text = pytesseract.image_to_string(image, lang="eng")
    elapsed = time.time() - start
    return normalize_text(text), elapsed


def run_easyocr(image):
    reader = get_easy_reader()
    start = time.time()
    results = reader.readtext(image, detail=0, paragraph=True)
    text = " ".join(results)
    elapsed = time.time() - start
    return normalize_text(text), elapsed



def evaluate_single_image(image_path, gt_text_path, use_detection=False):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    with open(gt_text_path, "r", encoding="utf-8") as f:
        gt_text = normalize_text(f.read())

    regions = detect_text_regions_placeholder(image) if use_detection else [image]

    # Tesseract
    tess_texts = []
    tess_time = 0.0
    for region in regions:
        text, t = run_tesseract(region)
        tess_texts.append(text)
        tess_time += t
    tess_text = normalize_text(" ".join(tess_texts))

    # EasyOCR / DL model
    dl_texts = []
    dl_time = 0.0
    for region in regions:
        text, t = run_easyocr(region)
        dl_texts.append(text)
        dl_time += t
    dl_text = normalize_text(" ".join(dl_texts))

    result = {
        "image_name": Path(image_path).name,
        "ground_truth": gt_text,

        "tesseract_text": tess_text,
        "tesseract_word_acc": word_level_accuracy(gt_text, tess_text),
        "tesseract_cer": character_error_rate(gt_text, tess_text),
        "tesseract_time_sec": tess_time,

        "dl_text": dl_text,
        "dl_word_acc": word_level_accuracy(gt_text, dl_text),
        "dl_cer": character_error_rate(gt_text, dl_text),
        "dl_time_sec": dl_time,
    }

    return result


# =========================
# =========================
def evaluate_dataset(images_dir, gt_dir, output_report="ocr_report.txt", use_detection=False):
    images_dir = Path(images_dir)
    gt_dir = Path(gt_dir)

    print("images_dir =", images_dir.resolve())
    print("gt_dir     =", gt_dir.resolve())
    print("images_dir exists =", images_dir.exists())
    print("gt_dir exists     =", gt_dir.exists())

    if not images_dir.exists():
        print("Error: images folder does not exist.")
        return

    if not gt_dir.exists():
        print("Error: ground truth folder does not exist.")
        return

    image_files = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
    ])

    print("Found images:", [p.name for p in image_files])

    if not image_files:
        print("No test images found.")
        return

    results = []

    for image_path in image_files:
        gt_path = gt_dir / f"{image_path.stem}.txt"

        print("\nProcessing:", image_path.name)
        print("Expected GT:", gt_path.name)
        print("GT exists   :", gt_path.exists())

        if not gt_path.exists():
            print(f"GT file missing for: {image_path.name}")
            continue

        try:
            result = evaluate_single_image(image_path, gt_path, use_detection=use_detection)
            results.append(result)
            print(f"Done: {image_path.name}")
            print(f"Tesseract Word Acc: {result['tesseract_word_acc']:.4f}")
            print(f"Tesseract CER     : {result['tesseract_cer']:.4f}")
            print(f"EasyOCR Word Acc  : {result['dl_word_acc']:.4f}")
            print(f"EasyOCR CER       : {result['dl_cer']:.4f}")
        except Exception as e:
            print(f"Error on {image_path.name}: {e}")

    print("\nResults count =", len(results))

    if not results:
        print("No valid results.")
        return

    avg_tess_word = sum(r["tesseract_word_acc"] for r in results) / len(results)
    avg_tess_cer = sum(r["tesseract_cer"] for r in results) / len(results)
    avg_tess_time = sum(r["tesseract_time_sec"] for r in results) / len(results)

    avg_dl_word = sum(r["dl_word_acc"] for r in results) / len(results)
    avg_dl_cer = sum(r["dl_cer"] for r in results) / len(results)
    avg_dl_time = sum(r["dl_time_sec"] for r in results) / len(results)

    with open(output_report, "w", encoding="utf-8") as f:
        f.write("OCR Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        for r in results:
            f.write(f"Image: {r['image_name']}\n")
            f.write(f"Ground Truth: {r['ground_truth']}\n\n")

            f.write("[Tesseract]\n")
            f.write(f"Predicted Text: {r['tesseract_text']}\n")
            f.write(f"Word Accuracy: {r['tesseract_word_acc']:.4f}\n")
            f.write(f"CER: {r['tesseract_cer']:.4f}\n")
            f.write(f"Time (sec): {r['tesseract_time_sec']:.4f}\n\n")

            f.write("[DL Model - EasyOCR]\n")
            f.write(f"Predicted Text: {r['dl_text']}\n")
            f.write(f"Word Accuracy: {r['dl_word_acc']:.4f}\n")
            f.write(f"CER: {r['dl_cer']:.4f}\n")
            f.write(f"Time (sec): {r['dl_time_sec']:.4f}\n")
            f.write("-" * 60 + "\n\n")

        f.write("FINAL AVERAGES\n")
        f.write("=" * 60 + "\n")
        f.write(f"Tesseract Avg Word Accuracy: {avg_tess_word:.4f}\n")
        f.write(f"Tesseract Avg CER: {avg_tess_cer:.4f}\n")
        f.write(f"Tesseract Avg Time: {avg_tess_time:.4f}\n\n")

        f.write(f"EasyOCR Avg Word Accuracy: {avg_dl_word:.4f}\n")
        f.write(f"EasyOCR Avg CER: {avg_dl_cer:.4f}\n")
        f.write(f"EasyOCR Avg Time: {avg_dl_time:.4f}\n")

    print("\nReport saved to:", Path(output_report).resolve())


# =========================
# =========================
if __name__ == "__main__":
    print("Evaluation started...")

    evaluate_dataset(
        images_dir="test_images",
        gt_dir="ground_truth",
        output_report="ocr_report.txt",
        use_detection=False
    )
