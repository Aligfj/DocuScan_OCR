"""
Smart Document Scanner — Phase 1: Traditional CV Pipeline
=========================================================
Dependencies:
    pip install opencv-python pytesseract numpy
"""

import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ASUS\Desktop\mini_projet\tesseract\tesseract.exe"

# ─────────────────────────────────────────────
# STEP 1 — Load & Preprocess
# ─────────────────────────────────────────────

def load_and_preprocess(image_path: str, max_dim: int = 800):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return original, gray, blurred


# ─────────────────────────────────────────────
# STEP 2 — Edge Detection
# ─────────────────────────────────────────────

def detect_edges(blurred: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(blurred, 75, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


# ─────────────────────────────────────────────
# STEP 3 — Document Quad Detection
# ─────────────────────────────────────────────

def find_document_corners(edges: np.ndarray,
                           original: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contours detected. Check image quality.")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    doc_corners = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_corners = approx
            break

    if doc_corners is None:
        print("[warn] No 4-corner contour found — using extreme-point fallback.")
        c = contours[0]
        left   = tuple(c[c[:, :, 0].argmin()][0])
        right  = tuple(c[c[:, :, 0].argmax()][0])
        top    = tuple(c[c[:, :, 1].argmin()][0])
        bottom = tuple(c[c[:, :, 1].argmax()][0])
        doc_corners = np.array([[left], [top], [right], [bottom]])

    return doc_corners


def draw_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    vis = image.copy()
    cv2.drawContours(vis, [corners], -1, (0, 255, 0), 2)
    for pt in corners.reshape(4, 2):
        cv2.circle(vis, tuple(pt), 6, (0, 0, 255), -1)
    return vis


# ─────────────────────────────────────────────
# STEP 4 — Perspective Warp
# ─────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def four_point_transform(image: np.ndarray,
                          corners: np.ndarray) -> np.ndarray:
    src = order_points(corners)
    (tl, tr, br, bl) = src

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    W = int(max(wA, wB))

    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    H = int(max(hA, hB))

    dst = np.array([
        [0,     0    ],
        [W - 1, 0    ],
        [W - 1, H - 1],
        [0,     H - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (W, H))

    return warped


# ─────────────────────────────────────────────
# STEP 5 — Image Enhancement + OCR
# ─────────────────────────────────────────────

def enhance(warped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    clean = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    clean = cv2.fastNlMeansDenoising(clean, h=10)
    return clean


def run_ocr(clean: np.ndarray, lang: str = "eng") -> str:
    config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(clean, lang=lang, config=config)
    return text.strip()


# ─────────────────────────────────────────────
# AFFICHAGE JUPYTER — remplace cv2.imshow
# ─────────────────────────────────────────────

def show(title: str, img: np.ndarray):
    """Affiche une image inline dans Jupyter."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.title(title)
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MAIN — exécution directe dans Jupyter
# ─────────────────────────────────────────────

IMAGE_PATH = r"C:\Users\ASUS\Desktop\mini_projet\plo.webp"  # ← ton image
LANG       = "eng"        # ← change en "eng+ara+fra" si besoin
DEBUG      = True         # ← True = affiche chaque étape dans Jupyter

print(f"\n[1/5] Chargement : {IMAGE_PATH}")
original, gray, blurred = load_and_preprocess(IMAGE_PATH)

print("[2/5] Détection des bords...")
edges = detect_edges(blurred)

print("[3/5] Recherche des coins du document...")
corners = find_document_corners(edges, original)

if DEBUG:
    show("Étape 2 — Bords (Canny)", edges)
    show("Étape 3 — Coins détectés", draw_corners(original, corners))

print("[4/5] Correction de perspective...")
warped = four_point_transform(original, corners)

if DEBUG:
    show("Étape 4 — Document redressé", warped)

print("[5/5] Amélioration de l'image + OCR...")
clean = enhance(warped)

if DEBUG:
    show("Étape 5 — Image nettoyée (avant OCR)", clean)

text = run_ocr(clean, lang=LANG)

print("\n" + "─" * 50)
print("TEXTE EXTRAIT")
print("─" * 50)
print(text)
print("─" * 50)
