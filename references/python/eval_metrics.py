from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from loguru import logger
from matplotlib.cm import Blues
from sklearn.metrics import ConfusionMatrixDisplay
from supervision import box_non_max_suppression

# Add this type alias
ClassGroupings = dict[str, list[str]]


@dataclass
class InferenceStats:
    """Statistics about the inference process."""

    total_predictions: int = 0
    skipped_invalid_boxes: int = 0
    skipped_invalid_categories: int = 0
    skipped_missing_images: int = 0
    skipped_unavailable_class: int = 0
    processed_images: int = 0
    empty_predictions: int = 0

    def print_summary(self) -> None:
        """Print a summary of the inference statistics."""
        logger.info("\nInference Statistics:")
        logger.info(f"Total raw predictions: {self.total_predictions}")
        logger.info(f"Processed images: {self.processed_images}")
        logger.info(f"Empty predictions: {self.empty_predictions}")
        logger.info(f"  - Invalid boxes: {self.skipped_invalid_boxes}")
        logger.info(f"  - Invalid categories: {self.skipped_invalid_categories}")
        logger.info(f"  - Missing images: {self.skipped_missing_images}")
        logger.info(f"  - Classes not in validation set: {self.skipped_unavailable_class}")
        valid_predictions = (
            self.total_predictions
            - self.skipped_invalid_boxes
            - self.skipped_invalid_categories
            - self.skipped_missing_images
            - self.skipped_unavailable_class
        )
        logger.info(f"Final valid predictions: {valid_predictions}")


def validate_bbox(bbox: list[Any]) -> bool:
    """
    Validate a bounding box format and values.

    Args:
        bbox: List containing [x, y, width, height]

    Returns:
        bool: True if bbox is valid, False otherwise
    """
    if len(bbox) != 4:
        return False

    # Check for reasonable coordinate values
    try:
        # Attempt to unpack and convert to float
        x, y, w, h = map(float, bbox)
    except (ValueError, TypeError):
        return False

    # Check for negative or zero dimensions
    if w <= 0 or h <= 0:
        return False

    # Check for NaN or infinite values
    if any(np.isnan(v) or np.isinf(v) for v in map(float, bbox)):
        return False

    return True


class Evaluator:
    eval_f_beta: dict[float, list[dict[str, Any]]]
    model_class_list: list[str]
    category_df: pl.DataFrame
    category_name_to_id: dict[str, int]
    images_df: pl.DataFrame
    gt_df: pl.DataFrame
    valid_categories: set[int]
    predictions_df: pl.DataFrame
    matched_predictions: list[Any]
    class_groupings: dict[str, list[str]] | None

    def __init__(
        self,
        annotations_df: pl.DataFrame,
        images_df: pl.DataFrame,
        predictions_df: pl.DataFrame,
        model_class_list: list[str],
        nms_iou_threshold: float = 0.5,
        class_groupings: dict[str, list[str]] | None = None,
    ) -> None:
        """
        Initialize evaluation object using Polars DataFrames.

        Args:
            annotations_df: DataFrame containing ground truth annotations
            images_df: DataFrame containing image metadata
            predictions_df: DataFrame containing model predictions with columns:
                - image_id: Integer ID of the image
                - class_name: Predicted class name
                - bbox: List of [x, y, width, height] in absolute pixels
                - score: Confidence score of the prediction
            model_class_list: List of class names the model can predict
            nms_iou_threshold: IoU threshold for non-maximum suppression
            class_groupings: Optional mapping of group names to class lists
        """

        # Log initial statistics
        logger.info("\nDataFrame Statistics:")
        logger.info(f"Total images: {len(images_df)}")
        logger.info(f"Total annotations: {len(annotations_df)}")

        # Create category DataFrame from unique classes in annotations
        self.model_class_list = list(model_class_list)
        unique_classes = annotations_df["class_name"].unique().to_list()
        self.category_df = pl.DataFrame({"id": range(len(unique_classes)), "name": unique_classes})
        self.category_name_to_id = dict(zip(self.category_df["name"], self.category_df["id"], strict=False))

        # Create image id mapping
        image_id_map = dict(zip(images_df["file_name"], images_df["image_id"], strict=False))

        # Store images_df for camera statistics
        self.images_df = images_df

        # Process ground truth DataFrame to create bbox and add image_id
        logger.info("\nProcessing ground truth annotations...")
        if annotations_df.is_empty():
            self.gt_df = pl.DataFrame(
                schema={
                    "image_id": pl.Int64,
                    "category_id": pl.Int64,
                    "bbox": pl.List(pl.Float64),
                }
            )
        else:
            self.gt_df = annotations_df.with_columns(
                [
                    # Create bbox list from individual coordinates
                    pl.concat_list(
                        [
                            pl.col("box_x1").cast(pl.Float64),
                            pl.col("box_y1").cast(pl.Float64),
                            pl.col("box_width").cast(pl.Float64),
                            pl.col("box_height").cast(pl.Float64),
                        ]
                    ).alias("bbox"),
                    # Look up image_id from file_name
                    pl.col("file_name").map_elements(lambda x: image_id_map.get(x)).alias("image_id"),
                    # Map class_name to category_id
                    pl.col("class_name").map_elements(lambda x: self.category_name_to_id.get(x)).alias("category_id"),
                ]
            ).select(["image_id", "category_id", "bbox"])

            logger.info(f"Example bbox format: {self.gt_df.select('bbox').head(1).item()}")
        logger.info(f"Processed {len(self.gt_df)} ground truth boxes")

        # Validate and process predictions
        self.valid_categories = set(self.category_df["id"].to_list())
        processed_dt = self._process_predictions(predictions_df)

        if len(processed_dt) == 0:
            self.predictions_df = pl.DataFrame(
                schema={
                    "image_id": pl.Int64,
                    "category_id": pl.Int64,
                    "bbox": pl.List(pl.Float64),
                    "score": pl.Float64,
                }
            )
        else:
            self.predictions_df = processed_dt

        # Initialize other attributes
        self.matched_predictions = []
        self.class_groupings = class_groupings
        # Will be initialized in _calculate_fbeta_over_thresholds
        self.eval_f_beta = {}

    def _process_predictions(self, predictions_df: pl.DataFrame) -> pl.DataFrame:
        """
        Process predictions and convert class names to category IDs.

        Args:
            predictions_df: DataFrame containing model predictions

        Returns:
            Processed DataFrame with validated predictions
        """
        valid_predictions: list[dict[str, Any]] = []
        stats = InferenceStats()

        for row in predictions_df.iter_rows(named=True):
            stats.total_predictions += 1

            # Check for empty predictions
            score = row.get("score", 0)
            bbox = row.get("bbox", [0, 0, 0, 0])
            assert isinstance(score, int | float), f"Score must be a number, got {type(score)}"

            if score == 0 and all(v == 0 for v in bbox):
                stats.empty_predictions += 1
                continue

            # Check for valid class name
            class_name = row.get("class_name")
            if not class_name:
                stats.skipped_invalid_categories += 1
                continue
            assert isinstance(class_name, str), f"Class name must be a string, got {type(class_name)}"

            # Check for valid category ID
            category_id = self.category_name_to_id.get(class_name)
            if category_id is None or category_id not in self.valid_categories:
                stats.skipped_invalid_categories += 1
                continue
            assert isinstance(category_id, int), f"Category ID must be an integer, got {type(category_id)}"

            # Check for valid image ID
            image_id = row.get("image_id")
            if not image_id:
                stats.skipped_missing_images += 1
                continue
            assert isinstance(image_id, int), f"Image ID must be an integer, got {type(image_id)}"

            # Check for valid bounding box
            if not validate_bbox(bbox):
                stats.skipped_invalid_boxes += 1
                continue

            valid_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": row["score"],
                }
            )

        stats.print_summary()
        return pl.DataFrame(valid_predictions)

    @staticmethod
    def extract_camera_id(frame_uri: str) -> str:
        """
        Extract camera ID from frame URI.

        Args:
            frame_uri: Path to the frame/image

        Returns:
            Camera ID as string

        Example URIs:
            gs://binit-machine-learning/images/2024-11-19T16:50:34.205469303Z-120.jpg
            gs://binit-machine-learning/images/2023-03-08 12:24:21.728873548 +0000 UTC m=+154973.734070408-24-frame1.png
        """
        # Extract the filename from the path and remove the extension
        filename = frame_uri.split("/")[-1].rsplit(".", 1)[0]

        # Handle cases with "frame" in the name
        if "frame" in filename.lower():
            # Get content before "frame" and extract last number
            pre_frame = filename.lower().split("frame")[0]
            # Extract last number sequence before "frame"
            camera_id = pre_frame.split("-")[-2]
        else:
            # Get the camera ID (last segment after the last hyphen)
            camera_id = filename.split("-")[-1]

        return camera_id

    def summarize_camera_statistics(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Summarize evaluation statistics by camera ID and class.
        """
        # Overall camera stats
        camera_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Per-class camera stats
        camera_class_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # Extract camera IDs from image filenames in images_df
        images_with_cameras = self.images_df.with_columns(pl.col("frame_uri").map_elements(self.extract_camera_id).alias("camera_id"))

        # Create mapping of image_id to camera_id and frame_uri
        image_id_to_frame_uri = dict(zip(images_with_cameras["image_id"], images_with_cameras["frame_uri"], strict=False))
        frame_uri_to_camera = dict(zip(images_with_cameras["frame_uri"], images_with_cameras["camera_id"], strict=False))

        # For each class with best threshold
        for class_id in self.category_df["id"]:
            class_name = self.category_df.filter(pl.col("id") == class_id).select("name").item()

            # Get results for this class
            class_results = [r for r in self.eval_f_beta[0.5] if r["class_id"] == class_id]
            if not class_results:
                continue

            # Get result with best F-beta score
            best_result = max(class_results, key=lambda x: x["fbeta"])
            conf_thr = best_result["conf_thr"]

            # Get predictions and ground truth for this class
            class_dt = self.predictions_df.filter((pl.col("category_id") == class_id) & (pl.col("score") >= conf_thr))
            class_gt = self.gt_df.filter(pl.col("category_id") == class_id)

            # Count FPs and TPs by camera using the actual metrics from evaluate_category
            tp = best_result["tp"]
            fp = best_result["fp"]
            fn = best_result["fn"]

            # Distribute these across cameras
            for row in class_dt.iter_rows(named=True):
                if row["image_id"] is not None:  # Skip placeholder predictions
                    frame_uri = image_id_to_frame_uri.get(row["image_id"])
                    camera_id = frame_uri_to_camera.get(frame_uri, "unknown")
                    # If we've used all our TPs, this must be an FP
                    if tp > 0:
                        camera_stats[camera_id]["tp"] += 1
                        camera_class_stats[camera_id][class_name]["tp"] += 1
                        tp -= 1
                    else:
                        camera_stats[camera_id]["fp"] += 1
                        camera_class_stats[camera_id][class_name]["fp"] += 1
                        fp -= 1

            # Distribute FNs across cameras that have ground truth but no matched predictions
            for row in class_gt.iter_rows(named=True):
                frame_uri = image_id_to_frame_uri.get(row["image_id"])
                camera_id = frame_uri_to_camera.get(frame_uri, "unknown")
                if fn > 0:
                    camera_stats[camera_id]["fn"] += 1
                    camera_class_stats[camera_id][class_name]["fn"] += 1
                    fn -= 1

        # Convert overall stats to DataFrame
        camera_records = [
            {
                "camera_id": camera_id,
                "true_positives": stats["tp"],
                "false_positives": stats["fp"],
                "false_negatives": stats["fn"],
                "precision": (stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0),
                "recall": (stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0),
            }
            for camera_id, stats in camera_stats.items()
        ]

        # Convert per-class stats to DataFrame
        camera_class_records: list[dict[str, Any]] = []
        for camera_id, class_stats in camera_class_stats.items():
            for class_name, stats in class_stats.items():
                camera_class_records.append(
                    {
                        "camera_id": camera_id,
                        "class_name": class_name,
                        "true_positives": stats["tp"],
                        "false_positives": stats["fp"],
                        "false_negatives": stats["fn"],
                        "precision": (stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0),
                        "recall": (stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0),
                    }
                )

        return (
            pl.DataFrame(camera_records).sort("camera_id"),
            pl.DataFrame(camera_class_records).sort(["camera_id", "class_name"]),
        )

    def calculate_metrics_at_best_thresholds(self, beta: float = 1, iou_threshold: float = 0.5) -> pl.DataFrame:
        """
        Find the confidence threshold that optimizes the F-beta score for each class.
        Just performs a grid sweep.

        Args:
            beta: Defines the trade-off between precision and recall.

        From the scikit-learn docs:
        - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
        The beta parameter represents the ratio of recall importance to precision importance.
        beta > 1 gives more weight to recall, while beta < 1 favors precision.
        For example, beta = 2 makes recall twice as important as precision, while beta = 0.5 does the opposite.
        Asymptotically, beta -> +inf considers only recall, and beta -> 0 only precision.
        """
        best_thresholds: list[dict[str, Any]] = []
        # Only used within the class
        metrics_over_thresholds = self._calculate_fbeta_over_thresholds(
            iou_threshold=iou_threshold,
            beta=beta,
        )

        for class_id in self.category_df["id"]:
            class_name = self.category_df.filter(pl.col("id") == class_id).select("name").item()
            if class_name not in self.model_class_list:
                continue

            logger.info(f"Processing thresholds for {class_name}")

            # Filter results for this class
            class_results = metrics_over_thresholds.filter(pl.col("class_id") == class_id)

            if not class_results.is_empty():
                # For classes with no ground truth, choose threshold that minimizes FP
                if all(class_results.select("tp").to_numpy().flatten() == 0) and all(class_results.select("fn").to_numpy().flatten() == 0):
                    # Find index of minimum FP and highest conf_thr when tied
                    min_fp = class_results.select("fp").min().item()
                    best_result = class_results.filter(pl.col("fp") == min_fp).sort("conf_thr", descending=True).row(0, named=True)
                    logger.info(f"Class {class_name} - optimizing for minimum FP")
                else:
                    # Find result with best F-beta score
                    max_fbeta = class_results.select("fbeta").max().item()
                    best_result = class_results.filter(pl.col("fbeta") == max_fbeta).row(0, named=True)

                best_thresholds.append(
                    {
                        "class_id": int(class_id),
                        "class_name": str(class_name),
                        "iou_thr": float(iou_threshold),
                        "best_fbeta": float(best_result["fbeta"]),
                        "conf": round(float(best_result["conf_thr"]), 2),  # Round to 2 decimal places
                        "precision": float(best_result["precision"]),
                        "recall": float(best_result["recall"]),
                        "tp": int(best_result["tp"]),
                        "fp": int(best_result["fp"]),
                        "fn": int(best_result["fn"]),
                    }
                )

        if not best_thresholds:
            logger.warning("No best thresholds found for any class")
            return pl.DataFrame(
                schema={
                    "class_id": pl.Int64,
                    "class_name": pl.Utf8,
                    "iou_thr": pl.Float64,
                    "best_fbeta": pl.Float64,
                    "conf": pl.Float64,
                    "precision": pl.Float64,
                    "recall": pl.Float64,
                    "tp": pl.Int64,
                    "fp": pl.Int64,
                    "fn": pl.Int64,
                }
            )

        results_df = pl.DataFrame(best_thresholds)
        logger.info(f"Found best thresholds for {len(results_df)} classes")
        assert len(results_df) > 0, "Expected non-empty DataFrame of best thresholds"

        evaluation_results = results_df.sort("class_name")

        # Validate the schema of the returned DataFrame
        expected_columns = {"class_id", "class_name", "iou_thr", "best_fbeta", "conf", "precision", "recall", "tp", "fp", "fn"}
        actual_columns = set(evaluation_results.columns)
        assert expected_columns.issubset(actual_columns), f"Missing columns in results. Expected {expected_columns}, got {actual_columns}"

        # Validate data types for key columns
        assert evaluation_results["class_id"].dtype == pl.Int64, f"class_id should be Int64, got {evaluation_results['class_id'].dtype}"
        assert evaluation_results["class_name"].dtype == pl.Utf8, f"class_name should be Utf8, got {evaluation_results['class_name'].dtype}"
        assert evaluation_results["best_fbeta"].dtype == pl.Float64, f"best_fbeta should be Float64, got {evaluation_results['best_fbeta'].dtype}"

        # Configure Polars to display full output
        with pl.Config(
            tbl_rows=-1,  # Show all rows
            tbl_cols=-1,  # Show all columns
            tbl_width_chars=None,  # No width limit
            fmt_str_lengths=100,  # Show full string content
            set_tbl_hide_dataframe_shape=True,  # Hide shape info for cleaner output
        ):
            print(f"\n=== EVALUATION RESULTS for beta value of {beta} ==")
            print(evaluation_results)
            print("=" * 50)

        return evaluation_results

    def _apply_nms(self, predictions_df: pl.DataFrame, iou_threshold: float) -> pl.DataFrame:
        """
        Apply NMS to predictions per category and image.

        Args:
            predictions_df: DataFrame containing predictions to apply NMS to
            iou_threshold: IoU threshold for non-maximum suppression

        Returns:
            DataFrame with filtered predictions after NMS
        """
        logger.info(f"Applying non-maximum suppression with IoU threshold {iou_threshold}")
        assert 0 <= iou_threshold <= 1, f"IoU threshold must be between 0 and 1, got {iou_threshold}"
        # Create a copy to store filtered results
        filtered_dfs: list[pl.DataFrame] = []

        # Process each category and image combination
        for cat_id in predictions_df["category_id"].unique():
            cat_preds = predictions_df.filter(pl.col("category_id") == cat_id)

            for img_id in cat_preds["image_id"].unique():
                if img_id is None:  # Skip placeholder predictions
                    continue

                img_preds = cat_preds.filter(pl.col("image_id") == img_id)

                if len(img_preds) == 0:
                    continue

                # Convert bbox format from [x, y, w, h] to [x1, y1, x2, y2]
                boxes = np.array(list(img_preds.select("bbox").to_numpy().flatten()))
                xyxy = np.column_stack(
                    [
                        boxes[:, 0],  # x1
                        boxes[:, 1],  # y1
                        boxes[:, 0] + boxes[:, 2],  # x2
                        boxes[:, 1] + boxes[:, 3],  # y2
                    ]
                )

                scores = img_preds.select("score").to_numpy().flatten()

                # Combine boxes and scores for NMS
                predictions = np.hstack((xyxy, scores.reshape(-1, 1)))

                # Apply NMS and get indices of kept boxes
                kept_mask = box_non_max_suppression(predictions=predictions, iou_threshold=iou_threshold)

                # Convert boolean mask to integer indices
                kept_indices = np.where(kept_mask)[0]

                # Keep only the selected predictions
                # kept_indices is a numpy array so it's guaranteed to have a length
                if len(kept_indices) > 0:
                    logger.debug(f"NMS kept {len(kept_indices)} out of {len(img_preds)} boxes for image {img_id}, category {cat_id}")
                    # First get the kept boxes in xyxy format
                    kept_boxes_xyxy = xyxy[kept_indices]

                    # Convert back to xywh format
                    kept_boxes_xywh = np.column_stack(
                        [
                            kept_boxes_xyxy[:, 0],  # x
                            kept_boxes_xyxy[:, 1],  # y
                            kept_boxes_xyxy[:, 2] - kept_boxes_xyxy[:, 0],  # width
                            kept_boxes_xyxy[:, 3] - kept_boxes_xyxy[:, 1],  # height
                        ]
                    )

                    # Create a new DataFrame with the kept predictions
                    kept_preds = pl.DataFrame(
                        {
                            "image_id": [img_preds.select("image_id").row(idx)[0] for idx in kept_indices],
                            "category_id": [img_preds.select("category_id").row(idx)[0] for idx in kept_indices],
                            "bbox": [box.tolist() for box in kept_boxes_xywh],
                            "score": [img_preds.select("score").row(idx)[0] for idx in kept_indices],
                        }
                    )

                    filtered_dfs.append(kept_preds)

        # Also include placeholder predictions for missing classes
        placeholder_preds = predictions_df.filter(pl.col("image_id").is_null())

        # Combine all filtered predictions and placeholders
        if filtered_dfs:
            if not placeholder_preds.is_empty():
                filtered_dfs.append(placeholder_preds)
            return pl.concat(filtered_dfs)
        elif not placeholder_preds.is_empty():
            return placeholder_preds
        return pl.DataFrame(schema=predictions_df.schema)

    def compute_iou_matrix(self, dt_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
        """
        Compute IoU matrix between two sets of boxes vectorized.
        Boxes are in format: [x, y, w, h]
        Returns matrix of shape (num_dt, num_gt)
        """
        logger.debug(f"Computing IoU matrix between {len(dt_boxes)} detection boxes and {len(gt_boxes)} ground truth boxes")
        if dt_boxes.size == 0 or gt_boxes.size == 0:
            logger.debug("Empty boxes array detected, returning empty IoU matrix")
            return np.array([])
        # Make sure inputs are the right shape
        assert dt_boxes.ndim == 2 and dt_boxes.shape[1] == 4, f"DT boxes shape is {dt_boxes.shape}"
        assert gt_boxes.ndim == 2 and gt_boxes.shape[1] == 4, f"GT boxes shape is {gt_boxes.shape}"

        # Convert arrays to float for division operation
        dt_boxes = dt_boxes.astype(np.float64)
        gt_boxes = gt_boxes.astype(np.float64)

        # Extract coordinates for easier reading
        dt_x = dt_boxes[:, 0]
        dt_y = dt_boxes[:, 1]
        dt_w = dt_boxes[:, 2]
        dt_h = dt_boxes[:, 3]

        gt_x = gt_boxes[:, 0]
        gt_y = gt_boxes[:, 1]
        gt_w = gt_boxes[:, 2]
        gt_h = gt_boxes[:, 3]

        # Compute intersection - using broadcasting for pairwise computations
        # For x-axis
        left = np.maximum(dt_x[:, None], gt_x[None, :])
        right = np.minimum(dt_x[:, None] + dt_w[:, None], gt_x[None, :] + gt_w[None, :])

        # For y-axis
        top = np.maximum(dt_y[:, None], gt_y[None, :])
        bottom = np.minimum(dt_y[:, None] + dt_h[:, None], gt_y[None, :] + gt_h[None, :])

        # Compute intersection area
        width = np.maximum(0, right - left)
        height = np.maximum(0, bottom - top)
        intersection = width * height

        # Compute areas
        dt_areas = dt_w * dt_h
        gt_areas = gt_w * gt_h

        # Compute union using broadcasting
        union = dt_areas[:, None] + gt_areas[None, :] - intersection

        # Create an output array of zeros with float64 dtype
        output = np.zeros(intersection.shape, dtype=np.float64)

        # Compute IoU
        iou = np.divide(intersection, union, out=output, where=union > 0)

        return iou

    def plot_confusion_matrix(self, filename: str) -> None:
        """
        Plot confusion matrix for all classes using sklearn's ConfusionMatrixDisplay.
        Uses the best confidence threshold results for each class.

        Args:
            filename: Where to save the plot
        """

        if self.eval_f_beta is None:
            logger.info("No evaluation results available. Run calculate_fbeta first.")
            return

        # Get best threshold results for each class at IoU 0.5
        best_results = []
        for class_id in self.category_df["id"]:
            class_results = [r for r in self.eval_f_beta[0.5] if r["class_id"] == class_id]
            if class_results:
                # Get result with best F-beta score
                best_result = max(class_results, key=lambda x: x["fbeta"])
                best_results.append(best_result)

        # Create DataFrame for easier handling
        results_df = pl.DataFrame(best_results)
        results_df = results_df.join(self.category_df.select(["id", "name"]), left_on="class_id", right_on="id")

        # Print debug info
        logger.info("\nBest threshold results used for confusion matrix:")
        for r in results_df.iter_rows(named=True):
            logger.info(f"Class {r['name']}: TP={r['tp']}, FP={r['fp']}, FN={r['fn']} (conf_thr={r['conf_thr']:.3f})")

        # Get class names in the same order as thresholding_df
        class_names = results_df.select("name").to_series().to_list()
        class_names.append("Background")  # Add background class

        n_classes = len(class_names) - 1  # Subtract 1 for background class
        cm = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

        # Fill confusion matrix using the values from best threshold results
        for idx, row in enumerate(results_df.iter_rows(named=True)):
            # True Positives on diagonal
            cm[idx, idx] = row["tp"]

            # False Positives in the background row (last row)
            cm[-1, idx] = row["fp"]

            # False Negatives in the background column (last column)
            cm[idx, -1] = row["fn"]

        # Print raw confusion matrix for debugging
        logger.info("\nRaw Confusion Matrix:")
        logger.info(cm)
        logger.info("Class names:", class_names)

        # Create plot with adjusted figure size
        fig_size = (
            max(12, 8 + 0.5 * (n_classes + 1)),  # width
            max(10, 6 + 0.5 * (n_classes + 1)),  # height
        )
        fig, ax = plt.subplots(figsize=fig_size)

        # Create and plot ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        disp.plot(
            ax=ax,
            cmap=Blues,
            values_format="d",  # Show as integers
            xticks_rotation="horizontal",  # Use "horizontal" to avoid rotation issues
            colorbar=True,
        )

        # Customize plot
        plt.title("Confusion Matrix (at Best Confidence Thresholds)")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(filename, bbox_inches="tight", dpi=300, pad_inches=0.5)
        plt.close()

    # Helper function to handle NaN values
    def np_nan_to_zeros(self, x: np.ndarray) -> np.ndarray:
        """
        Convert NaN values to zeros in a numpy array.
        """
        return np.where(np.isnan(x), 0, x)

    def calculate_metrics_at_current_thresholds(self, beta: float = 1, iou_thr: float = 0.5) -> pl.DataFrame:
        """
        Calculate F-beta scores for the provided predictions.

        This will take into account any thresholding which has already occured

        Args:
            beta: The beta parameter for F-beta score
            iou_thr: IoU threshold for matching predictions to ground truth

        Returns:
            pl.DataFrame: DataFrame containing evaluation metrics for each class
        """
        logger.info(f"Calculating F-beta scores with beta={beta} at IoU threshold {iou_thr}")
        assert beta > 0, f"Beta must be positive, got {beta}"
        results: list[dict[str, Any]] = []

        # If there are no predictions, return an empty DataFrame
        if len(self.predictions_df) == 0:
            return pl.DataFrame(
                schema={
                    "class_id": pl.Int64,
                    "class_name": pl.Utf8,
                    "precision": pl.Float64,
                    "recall": pl.Float64,
                    "fbeta": pl.Float64,
                    "tp": pl.Int64,
                    "fp": pl.Int64,
                    "fn": pl.Int64,
                }
            )

        # For each class
        for class_id in self.category_df["id"]:
            class_name = self.category_df.filter(pl.col("id") == class_id).select("name").item()
            if class_name not in self.model_class_list:
                continue

            # Get predictions and ground truth for this class
            class_dt = self.predictions_df.filter(pl.col("category_id") == class_id)
            class_gt = self.gt_df.filter(pl.col("category_id") == class_id)

            # Calculate metrics
            tp, fp, fn = self.evaluate_category(class_dt, class_gt, iou_thr)

            logger.debug(f"Class {class_name}, TP={tp}, FP={fp}, FN={fn}")

            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            assert 0 <= precision <= 1, f"Precision must be between 0 and 1, got {precision}"
            assert 0 <= recall <= 1, f"Recall must be between 0 and 1, got {recall}"

            # Calculate F-beta score
            fbeta = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall) if precision + recall > 0 else 0

            results.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )

        # Convert results to Polars DataFrame
        evaluation_results = pl.DataFrame(results).sort("class_name")

        # Configure Polars to display full output
        with pl.Config(
            tbl_rows=-1,  # Show all rows
            tbl_cols=-1,  # Show all columns
            tbl_width_chars=None,  # No width limit
            fmt_str_lengths=100,  # Show full string content
            set_tbl_hide_dataframe_shape=True,  # Hide shape info for cleaner output
        ):
            print("\n=== EVALUATION RESULTS ===")
            print(evaluation_results)
            print("=" * 50)

        return evaluation_results

    def _calculate_fbeta_over_thresholds(
        self,
        beta: float = 1,
        conf_thresholds: np.ndarray | None = None,
        iou_threshold: float = 0.5,
    ) -> pl.DataFrame:
        """
        Calculate F-beta scores for different confidence thresholds.

        Args:
            beta: The beta parameter for F-beta score
            conf_thresholds: Optional array of confidence thresholds to evaluate
            iou_threshold: IoU threshold for matching predictions to ground truth

        Returns:
            pl.DataFrame: DataFrame containing evaluation metrics for each class at different thresholds
        """
        logger.info(f"Calculating F-beta scores with beta={beta}")
        assert beta > 0, f"Beta must be positive, got {beta}"
        # Initialize storage for evaluation results
        self.eval_f_beta = {}

        # Use provided thresholds or default to 101 evenly spaced values
        if conf_thresholds is None:
            conf_thresholds = np.linspace(0.0, 1.0, 101)

        results: list[dict[str, Any]] = []

        # For each class
        for class_id in self.category_df["id"]:
            class_name = self.category_df.filter(pl.col("id") == class_id).select("name").item()
            if class_name not in self.model_class_list:
                continue

            logger.info(f"\nProcessing {class_name}")

            # Get predictions and ground truth for this class
            class_dt = self.predictions_df.filter(pl.col("category_id") == class_id)
            class_gt = self.gt_df.filter(pl.col("category_id") == class_id)

            # Calculate metrics at different confidence thresholds
            for conf_thr in conf_thresholds:
                # Filter predictions by confidence threshold
                dt = class_dt.filter(pl.col("score") >= conf_thr)

                # Calculate metrics
                tp, fp, fn = self.evaluate_category(dt, class_gt, iou_threshold)

                logger.debug(f"Class {class_name}, conf={conf_thr:.2f}: TP={tp}, FP={fp}, FN={fn}")

                # Calculate precision and recall
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                assert 0 <= precision <= 1, f"Precision must be between 0 and 1, got {precision}"
                assert 0 <= recall <= 1, f"Recall must be between 0 and 1, got {recall}"

                # Calculate F-beta score
                fbeta = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall) if precision + recall > 0 else 0

                results.append(
                    {
                        "class_id": class_id,
                        "conf_thr": conf_thr,
                        "precision": precision,
                        "recall": recall,
                        "fbeta": fbeta,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                )

        return pl.DataFrame(results)

    def evaluate_category(self, predictions_df: pl.DataFrame, gt_df: pl.DataFrame, iou_threshold: float) -> tuple[int, int, int]:
        """
        Evaluate predictions for a single category.

        Args:
            predictions_df: DataFrame with predictions for a specific category
            gt_df: DataFrame with ground truth for a specific category
            iou_threshold: IoU threshold for matching predictions to ground truth

        Returns:
            Tuple of (true positives, false positives, false negatives)
        """
        logger.debug(f"Evaluating category with {len(predictions_df)} predictions and {len(gt_df)} ground truth boxes")
        assert 0 <= iou_threshold <= 1, f"IoU threshold must be between 0 and 1, got {iou_threshold}"
        tp = fp = fn = 0

        # If no ground truth, all predictions are false positives
        if len(gt_df) == 0:
            return 0, len(predictions_df), 0

        # Get unique image IDs from both dataframes
        gt_image_ids = set(gt_df["image_id"].unique())
        pred_image_ids = set(predictions_df["image_id"].unique())
        all_image_ids = gt_image_ids.union(pred_image_ids)

        # Process each image
        for image_id in all_image_ids:
            # Get ground truth and predictions for this image
            img_gt = gt_df.filter(pl.col("image_id") == image_id)
            img_dt = predictions_df.filter(pl.col("image_id") == image_id)

            if len(img_gt) == 0:
                # All predictions for this image are false positives
                fp += len(img_dt)
                continue

            if len(img_dt) == 0:
                # All ground truth for this image are false negatives
                fn += len(img_gt)
                continue

            # Convert boxes to numpy arrays without list comprehensions
            gt_boxes = np.array(list(img_gt.select("bbox").to_numpy().flatten()))
            dt_boxes = np.array(list(img_dt.select("bbox").to_numpy().flatten()))

            # Compute IoU matrix
            iou_matrix = self.compute_iou_matrix(dt_boxes, gt_boxes)

            if len(iou_matrix) == 0:
                fp += len(img_dt)
                fn += len(img_gt)
                continue

            # Find matches using IoU threshold
            matched_gt = set()
            for dt_idx in range(len(img_dt)):
                best_iou = 0
                best_gt_idx = -1

                for gt_idx in range(len(img_gt)):
                    if gt_idx in matched_gt:
                        continue

                    iou = iou_matrix[dt_idx, gt_idx]
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            # Count unmatched ground truth as false negatives
            fn += len(img_gt) - len(matched_gt)

        return tp, fp, fn

    def calculate_fbeta_data(self) -> dict[str, dict[str, np.ndarray]]:
        """
        Calculate F-beta curve data.
        """
        if self.eval_f_beta is None:
            logger.info("No evaluation results available. Run calculate_fbeta first.")
            return {}

        # Process each class
        fbeta_data: dict[str, dict[str, np.ndarray]] = {}
        for class_id in self.category_df["id"]:
            class_name = self.category_df.filter(pl.col("id") == class_id).select("name").item()
            if class_name not in self.model_class_list:
                continue

            # Get results for this class at IoU 0.5
            class_results = [r for r in self.eval_f_beta[0.5] if r["class_id"] == class_id]
            if not class_results:
                continue

            # Sort by confidence threshold
            class_results.sort(key=lambda x: x["conf_thr"])

            # Extract confidence thresholds and F-beta scores
            conf_thrs = np.array([r["conf_thr"] for r in class_results])
            fbeta_scores = np.array([r["fbeta"] for r in class_results])

            fbeta_data[class_name] = {"conf_thrs": conf_thrs, "fbeta": fbeta_scores}

        return fbeta_data

    def calculate_confusion_matrix(self) -> np.ndarray:
        """
        Calculate confusion matrix data.
        """
        if self.eval_f_beta is None:
            logger.info("No evaluation results available. Run calculate_fbeta first.")
            return np.array([])

        # Get best threshold results for each class at IoU 0.5
        best_results: list[dict[str, Any]] = []
        for class_id in self.category_df["id"]:
            class_results = [r for r in self.eval_f_beta[0.5] if r["class_id"] == class_id]
            if class_results:
                # Get result with best F-beta score
                best_result = max(class_results, key=lambda x: x["fbeta"])
                best_results.append(best_result)

        # Create DataFrame for easier handling
        results_df = pl.DataFrame(best_results)
        results_df = results_df.join(self.category_df.select(["id", "name"]), left_on="class_id", right_on="id")

        # Get class names in the same order as results
        class_names = results_df.select("name").to_series().to_list()
        class_names.append("Background")  # Add background class

        n_classes = len(class_names) - 1  # Subtract 1 for background class
        cm = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

        # Fill confusion matrix using the values from best threshold results
        for idx, row in enumerate(results_df.iter_rows(named=True)):
            # True Positives on diagonal
            cm[idx, idx] = row["tp"]
            # False Positives in the background row (last row)
            cm[-1, idx] = row["fp"]
            # False Negatives in the background column (last column)
            cm[idx, -1] = row["fn"]

        return cm
