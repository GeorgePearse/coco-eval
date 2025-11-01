import os

import polars as pl


def convert_offline_inference_preds_to_predictions_df(
    items: pl.DataFrame,
    filename_to_id: dict[str, int],
):
    """Convert inference response to a Polars DataFrame for evaluation.
    Provides coordinates as separate columns to match evaluator's expected format."""
    predictions = []
    for item in items.iter_rows(named=True):
        num_preds = len(item["detections.x"])
        for idx in range(num_preds):
            if item["detections.filtered_by"][idx] == "" or (item["detections.filtered_by"][idx] == "replaced_by_classifier"):
                # Convert normalized coordinates to absolute pixels
                x1_abs = float(item["detections.x"][idx]) * item["width"]
                y1_abs = float(item["detections.y"][idx]) * item["height"]

                # W and H are already width and height in normalized space
                width_abs = float(item["detections.w"][idx] - item["detections.x"][idx]) * item["width"]
                height_abs = float(item["detections.h"][idx] - item["detections.y"][idx]) * item["height"]

                segmentation_points = []
                for point in item["detections.segmentation_points"][idx]:
                    segmentation_points.append([point["x"], point["y"]])

                # Only add prediction if width and height are non-zero
                if width_abs > 0 and height_abs > 0:
                    predictions.append(
                        {
                            "file_name": os.path.basename(item["frame_uri"]),
                            "image_id": filename_to_id[os.path.basename(item["frame_uri"])],
                            "class_name": item["inference_category_name"][idx],
                            "score": item["detections.score"][idx],
                            "bbox": [x1_abs, y1_abs, width_abs, height_abs],
                            "segmentation_points": segmentation_points,
                            "pipeline_name": item["pipeline_name"],
                            "git_hash": item["git_hash"],
                        }
                    )

    return pl.DataFrame(predictions)
