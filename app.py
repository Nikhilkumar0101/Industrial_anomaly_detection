import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from src.model import PatchAnomalyDetector


def make_overlay(image: Image.Image, anomaly_map: np.ndarray):
    original = np.asarray(
        image.convert("RGB").resize(
            (anomaly_map.shape[1], anomaly_map.shape[0])
        )
    )

    anomaly_map = anomaly_map.astype(np.float32)
    low = float(anomaly_map.min())
    high = float(anomaly_map.max())

    if high > low:
        normalized = (
            (anomaly_map - low) / (high - low) * 255
        ).astype(np.uint8)
    else:
        normalized = np.zeros_like(anomaly_map, dtype=np.uint8)

    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.62, heatmap, 0.38, 0)

    return Image.fromarray(overlay), Image.fromarray(heatmap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Demo anomaly-score threshold; validate before real use.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public temporary Gradio link.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1.")

    # Bottle-only model.
    model_path = Path("outputs/model.pt")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Bottle model not found: {model_path}. "
            "Please train the Bottle model first."
        )

    detector = PatchAnomalyDetector.load(model_path)
    print(f"Loaded Bottle model: {model_path}")

    def predict(image):
        if image is None:
            return None, None, "Please upload a product image."

        score, anomaly_map = detector.predict_image(image)
        overlay, heat = make_overlay(image, anomaly_map)

        if score >= args.threshold:
            label = "ANOMALY DETECTED"
            explanation = (
                "The score is at or above the demo threshold."
            )
        else:
            label = "NORMAL"
            explanation = (
                "The score is below the demo threshold."
            )

        result = (
            f"Inspection decision: {label}\n"
            "Selected category: Bottle\n"
            f"Anomaly score: {score:.5f}\n"
            f"Demo threshold: {args.threshold:.2f}\n\n"
            f"{explanation}\n\n"
            "Important: This threshold is for demonstration only "
            "and has not been validated for production use. The anomaly "
            "score is not a calibrated probability. Results may be "
            "unreliable for images that differ from the training data. "
            "Review the image and heatmap; do not use this prototype "
            "as an autonomous quality-control decision."
        )

        return overlay, heat, result

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(
                type="pil",
                label="Upload bottle image",
            ),
        ],
        outputs=[
            gr.Image(type="pil", label="Anomaly overlay"),
            gr.Image(type="pil", label="Heatmap"),
            gr.Textbox(label="Inspection result"),
        ],
        title="Industrial Anomaly Inspector",
        description=(
            "Upload a bottle image to inspect it for potential "
            "anomalies. The model was trained on normal bottle images."
        ),
        examples=None,
    )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()