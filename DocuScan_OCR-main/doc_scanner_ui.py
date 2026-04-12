from __future__ import annotations

import cv2
import gradio as gr
import pandas as pd
import pytesseract

from evaluation import (
    compare_ocr_on_image,
    configure_tesseract,
    save_single_report,
)


DEFAULT_TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -----------------------------
# UI helpers
# -----------------------------
def build_metrics(result: dict, has_ground_truth: bool) -> pd.DataFrame:
    if has_ground_truth:
        rows = [
            {
                "Model": "Tesseract",
                "Word Accuracy": round(result["tesseract_word_acc"], 4),
                "CER": round(result["tesseract_cer"], 4),
                "Time (s)": round(result["tesseract_time_sec"], 4),
                "Characters": len(result["tesseract_text"]),
                "Words": len(result["tesseract_text"].split()),
            },
            {
                "Model": "EasyOCR",
                "Word Accuracy": round(result["dl_word_acc"], 4),
                "CER": round(result["dl_cer"], 4),
                "Time (s)": round(result["dl_time_sec"], 4),
                "Characters": len(result["dl_text"]),
                "Words": len(result["dl_text"].split()),
            },
        ]
    else:
        rows = [
            {
                "Model": "Tesseract",
                "Time (s)": round(result["tesseract_time_sec"], 4),
                "Characters": len(result["tesseract_text"]),
                "Words": len(result["tesseract_text"].split()),
            },
            {
                "Model": "EasyOCR",
                "Time (s)": round(result["dl_time_sec"], 4),
                "Characters": len(result["dl_text"]),
                "Words": len(result["dl_text"].split()),
            },
        ]

    return pd.DataFrame(rows)



def build_summary(result: dict, tesseract_lang: str, easy_langs: list[str], has_ground_truth: bool) -> str:
    if result["found_document"]:
        detection_text = "Document detected and warped."
    else:
        detection_text = "Document was not detected clearly. The full image was used."

    if has_ground_truth:
        metrics_text = "Ground truth was used for Word Accuracy and CER."
    else:
        metrics_text = "No ground truth was provided. Only OCR text and runtime are shown."

    return (
        f"Status: {detection_text}\n"
        f"Detection result: {result['detection_status']}\n"
        f"Tesseract path: {result['tesseract_path'] or 'Not set'}\n"
        f"Tesseract language: {tesseract_lang}\n"
        f"EasyOCR languages: {', '.join(easy_langs)}\n"
        f"Metrics: {metrics_text}"
    )



def choose_best_model(metrics_df: pd.DataFrame, has_ground_truth: bool) -> str:
    if metrics_df.empty:
        return "No result available"

    if not has_ground_truth:
        best_index = metrics_df["Time (s)"].astype(float).idxmin()
        return f"Fastest model: {metrics_df.loc[best_index, 'Model']}"

    best_acc_index = metrics_df["Word Accuracy"].astype(float).idxmax()
    best_cer_index = metrics_df["CER"].astype(float).idxmin()

    acc_model = metrics_df.loc[best_acc_index, "Model"]
    cer_model = metrics_df.loc[best_cer_index, "Model"]

    if acc_model == cer_model:
        return f"Best overall result: {acc_model}"

    return f"Best word accuracy: {acc_model}. Best CER: {cer_model}."



def process_image(
    image,
    ground_truth: str,
    tesseract_lang: str,
    tesseract_config: str,
    easyocr_langs: str,
    tesseract_cmd: str,
    blur_kernel: int,
    canny_low: int,
    canny_high: int,
    denoise_strength: int,
    threshold_block_size: int,
    threshold_c: int,
):
    if image is None:
        raise gr.Error("Please upload an image first.")

    tesseract_cmd = tesseract_cmd.strip() if tesseract_cmd else None
    configure_tesseract(tesseract_cmd)

    image_rgb = image.astype("uint8")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    easy_lang_list = [lang.strip() for lang in easyocr_langs.split(",") if lang.strip()]
    if not easy_lang_list:
        easy_lang_list = ["en"]

    try:
        result = compare_ocr_on_image(
            image_bgr,
            ground_truth=ground_truth,
            tesseract_cmd=tesseract_cmd,
            tesseract_lang=tesseract_lang.strip() or "eng",
            tesseract_config=tesseract_config.strip() or "--psm 6",
            easyocr_langs=easy_lang_list,
            blur_kernel=blur_kernel,
            canny_low=canny_low,
            canny_high=canny_high,
            denoise_strength=denoise_strength,
            threshold_block_size=threshold_block_size,
            threshold_c=threshold_c,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise gr.Error("Tesseract was not found. Check the executable path.") from exc
    except pytesseract.TesseractError as exc:
        raise gr.Error(f"Tesseract failed: {exc}") from exc
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    has_ground_truth = bool(result["ground_truth"])
    metrics_df = build_metrics(result, has_ground_truth)
    summary = build_summary(result, tesseract_lang, easy_lang_list, has_ground_truth)
    best_model = choose_best_model(metrics_df, has_ground_truth)
    report_path = save_single_report(
        result,
        tesseract_lang=tesseract_lang,
        easyocr_langs=easy_lang_list,
    )

    return (
        image_rgb,
        result["gray"],
        result["blurred"],
        result["edges"],
        result["document_box"][:, :, ::-1],
        result["warped"][:, :, ::-1],
        result["enhanced"],
        summary,
        best_model,
        result["tesseract_text"],
        result["dl_text"],
        metrics_df,
        report_path,
    )



def clear_outputs():
    return (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        "No result yet.",
        "No result yet.",
        "",
        "",
        pd.DataFrame(),
        None,
    )


CSS = """
.gradio-container {
    max-width: 1600px !important;
    padding: 0 16px 28px 16px !important;
}

.app-title h1 {
    margin-bottom: 6px !important;
}

.app-subtitle {
    color: #5b6470;
    margin-top: 0 !important;
}

.panel {
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 14px;
    background: white;
}

.section-label {
    font-size: 1.02rem;
    font-weight: 600;
    margin: 0 0 8px 0;
}

.compact-box textarea,
.compact-box input {
    font-size: 0.95rem !important;
}

.image-card {
    min-height: 280px;
}

.footer-note {
    color: #6b7280;
    font-size: 0.92rem;
}

@media (max-width: 1100px) {
    .gradio-container {
        padding: 0 10px 22px 10px !important;
    }
    .panel {
        padding: 12px;
    }
}

@media (max-width: 768px) {
    .app-title h1 {
        font-size: 1.6rem !important;
    }
    .section-label {
        font-size: 1rem;
    }
}
"""


with gr.Blocks(title="Document Scanner OCR", theme=gr.themes.Soft(), css=CSS) as demo:
    with gr.Column(elem_classes="app-title"):
        gr.Markdown("# Document Scanner OCR")
        gr.Markdown(
            "Upload a document photo, run the scanner pipeline, and compare Tesseract with EasyOCR.",
            elem_classes="app-subtitle",
        )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=360, elem_classes="panel"):
            gr.Markdown("Input", elem_classes="section-label")
            image_input = gr.Image(type="numpy", label="Document Image", height=360)
            ground_truth_input = gr.Textbox(
                label="Ground Truth Text",
                lines=8,
                placeholder="Paste the expected text here if you want Word Accuracy and CER.",
                elem_classes="compact-box",
            )

            with gr.Row():
                run_button = gr.Button("Run OCR", variant="primary")
                clear_button = gr.Button("Clear")

            with gr.Accordion("OCR Settings", open=True):
                with gr.Row():
                    tesseract_lang = gr.Textbox(label="Tesseract Language", value="eng")
                    easyocr_langs = gr.Textbox(label="EasyOCR Languages", value="en")
                tesseract_config = gr.Textbox(label="Tesseract Config", value="--psm 6")
                tesseract_cmd = gr.Textbox(label="Tesseract Path", value=DEFAULT_TESSERACT_PATH)

            with gr.Accordion("Scanner Settings", open=False):
                blur_kernel = gr.Slider(3, 15, value=5, step=2, label="Blur Kernel")
                with gr.Row():
                    canny_low = gr.Slider(20, 250, value=75, step=1, label="Canny Low")
                    canny_high = gr.Slider(20, 300, value=200, step=1, label="Canny High")
                denoise_strength = gr.Slider(1, 60, value=20, step=1, label="Denoise Strength")
                with gr.Row():
                    threshold_block_size = gr.Slider(3, 51, value=11, step=2, label="Threshold Block Size")
                    threshold_c = gr.Slider(-10, 20, value=2, step=1, label="Threshold C")

        with gr.Column(scale=4, min_width=360, elem_classes="panel"):
            gr.Markdown("Results", elem_classes="section-label")
            summary_output = gr.Textbox(label="Summary", lines=6, elem_classes="compact-box")
            best_model_output = gr.Textbox(label="Best Result", lines=2, elem_classes="compact-box")
            metrics_output = gr.Dataframe(label="Metrics", wrap=True, interactive=False)
            report_output = gr.File(label="Report File")
            gr.Markdown(
                "Tip: on desktop, keep the input panel on the left and review the pipeline tabs on the right after each run.",
                elem_classes="footer-note",
            )

    with gr.Tabs():
        with gr.TabItem("Pipeline"):
            with gr.Row(equal_height=False):
                with gr.Column(min_width=240):
                    original_output = gr.Image(label="Original", height=260, elem_classes="image-card")
                with gr.Column(min_width=240):
                    gray_output = gr.Image(label="Gray", height=260, elem_classes="image-card")
                with gr.Column(min_width=240):
                    blurred_output = gr.Image(label="Blurred", height=260, elem_classes="image-card")
                with gr.Column(min_width=240):
                    edges_output = gr.Image(label="Edges", height=260, elem_classes="image-card")

            with gr.Row(equal_height=False):
                with gr.Column(min_width=300):
                    detected_output = gr.Image(label="Detected Document", height=300, elem_classes="image-card")
                with gr.Column(min_width=300):
                    warped_output = gr.Image(label="Warped", height=300, elem_classes="image-card")
                with gr.Column(min_width=300):
                    enhanced_output = gr.Image(label="Enhanced", height=300, elem_classes="image-card")

        with gr.TabItem("OCR Text"):
            with gr.Row(equal_height=False):
                with gr.Column(min_width=360):
                    tesseract_text_output = gr.Textbox(label="Tesseract Text", lines=18)
                with gr.Column(min_width=360):
                    easyocr_text_output = gr.Textbox(label="EasyOCR Text", lines=18)

        with gr.TabItem("Instructions"):
            gr.Markdown(
                """
                1. Upload an image.
                2. Click Run OCR.
                3. Check the detected document, warped image, and enhanced image.
                4. Compare the Tesseract and EasyOCR outputs.
                5. Add ground truth text if you want Word Accuracy and CER.
                """
            )

    run_button.click(
        fn=process_image,
        inputs=[
            image_input,
            ground_truth_input,
            tesseract_lang,
            tesseract_config,
            easyocr_langs,
            tesseract_cmd,
            blur_kernel,
            canny_low,
            canny_high,
            denoise_strength,
            threshold_block_size,
            threshold_c,
        ],
        outputs=[
            original_output,
            gray_output,
            blurred_output,
            edges_output,
            detected_output,
            warped_output,
            enhanced_output,
            summary_output,
            best_model_output,
            tesseract_text_output,
            easyocr_text_output,
            metrics_output,
            report_output,
        ],
    )

    clear_button.click(
        fn=clear_outputs,
        outputs=[
            original_output,
            gray_output,
            blurred_output,
            edges_output,
            detected_output,
            warped_output,
            enhanced_output,
            summary_output,
            best_model_output,
            tesseract_text_output,
            easyocr_text_output,
            metrics_output,
            report_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
