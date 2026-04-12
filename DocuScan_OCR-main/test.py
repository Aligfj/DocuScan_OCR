import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import easyocr as easyocr_module
reader = easyocr_module.Reader(['en'], gpu=False)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def reorder(vertices):
    vertices = np.array(vertices, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = vertices.sum(axis=1)
    rect[0] = vertices[np.argmin(s)]   # top-left
    rect[2] = vertices[np.argmax(s)]   # bottom-right

    diff = np.diff(vertices, axis=1)
    rect[1] = vertices[np.argmin(diff)]  # top-right
    rect[3] = vertices[np.argmax(diff)]  # bottom-left

    return rect


def img_processing(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blurred, threshold1=140, threshold2=170)
    denoised = cv2.fastNlMeansDenoising(blurred, None, 30, 7, 21)
    clean = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return image, gray, blurred, edges , clean


def find_vertices(edges):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    vertices = None
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            vertices = approx.reshape(4, 2)
            break

    if vertices is None:
        h, w = edges.shape[:2]
        vertices = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    return reorder(vertices)


def crop_out(image, vertices, width=600, height=850):
    target = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    transform = cv2.getPerspectiveTransform(vertices, target)
    cropped = cv2.warpPerspective(image, transform, (width, height))

    return cropped


def enhance_for_ocr(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    clean = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return clean


# ===== Example usage =====
image_path = "test_images/1000097140(1).jpg"

image, gray, blurred, edges ,clean = img_processing(image_path)
vertices = find_vertices(edges)
warped = crop_out(image, vertices)


# draw contour points on original image
image_with_points = image.copy()
for x, y in vertices.astype(int):
    cv2.circle(image_with_points, (x, y), 10, (0, 255, 0), -1)

# display
plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(edges, cmap="gray")
plt.title("Edges")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(image_with_points, cv2.COLOR_BGR2RGB))
plt.title("Detected Vertices")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.title("Warped")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(clean, cmap="gray")
plt.title("Adaptive Threshold")
plt.axis("off")

plt.tight_layout()

plt.savefig("result.png", dpi=300)




def tesseract(img, lang_):
    text = pytesseract.image_to_string(img, lang=lang_)
    return text

print(tesseract(clean ,"eng"))
print("------------------------------------------")

import time
def run_easyocr(image):
    result = reader.readtext(image)
    return result

def extract_text(result, conf_threshold=0.5):
    lines = []

    for item in result:
        text = item[1]
        conf = item[2]

        if conf > conf_threshold:
            lines.append(text)

    return "\n".join(lines)
raw = run_easyocr(clean)
clean_text = extract_text(raw)

print(clean_text)
