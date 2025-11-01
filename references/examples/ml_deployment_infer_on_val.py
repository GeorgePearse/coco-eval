# pyright: ignore[reportAttributeAccessIssue]
import os
import random
from datetime import datetime

import polars as pl
import typer
import yaml
from data_management.training_split import TrainingSplit
from evaluation.eval_metrics import Evaluator, InferenceStats, validate_bbox
from ml_env_config.env import env
from PIL import Image
from slack_bot import send_file_to_slack
from tqdm import tqdm

from vision.deployment.config.inference import get_inference_config_by_name
from vision.mmdet.inferencer import CustomDetInferencer
from vision.tools.logger import logger  # pyright: ignore[reportAttributeAccessIssue]
from vision.tools.predictions import Prediction
from vision.tools.reporting import ModelEvaluationReport, expand_bbox
from vision.tools.trained_model import TrainedModel


class EvalMetrics:
    """Wrapper for metrics DataFrame with class and group metrics attributes."""

    def __init__(self, class_metrics: pl.DataFrame, group_metrics: pl.DataFrame | None = None):
        """Initialize with class and optional group metrics DataFrames."""
        self.class_metrics = class_metrics
        self.group_metrics = group_metrics


def create_prediction_df(predictions: list, image_ids: dict[str, int], stats: InferenceStats) -> list[dict]:
    """
    Create a prediction DataFrame with validation.

    Args:
        predictions: List of predictions
        image_ids: Dictionary mapping image filenames to image IDs
        stats: Statistics tracking object

    Returns:
        List of valid prediction dictionaries
    """
    valid_predictions = []

    for pred in predictions:
        stats.total_predictions += 1

        if pred.is_empty():
            stats.empty_predictions += 1
            continue

        file_name = os.path.basename(pred.file_path)
        image_id = image_ids.get(file_name)
        if image_id is None:
            stats.skipped_missing_images += 1
            logger.info(f"Warning: No image_id found for {pred.file_path}")
            continue

        x1, y1, x2, y2 = pred.box
        width = x2 - x1
        height = y2 - y1
        bbox = [x1, y1, width, height]

        if not validate_bbox(bbox):
            stats.skipped_invalid_boxes += 1
            logger.info(f"Warning: Invalid bbox {bbox} for image {pred.file_path}")
            continue

        # Get the class name from the prediction's class list
        class_name = pred.class_list[pred.label] if 0 <= pred.label < len(pred.class_list) else None
        if class_name is None:
            stats.skipped_invalid_categories += 1
            continue

        valid_predictions.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "label": pred.label,
                "class_name": class_name,
                "bbox": bbox,
                "score": pred.score,
            }
        )

    return valid_predictions


def run_inference(
    inferencer,
    file_paths: list[str],
    image_ids: dict[str, int],
    model_class_list: list[str],
    inference_kwargs: dict,
) -> pl.DataFrame:
    """Run inference and return processed predictions DataFrame."""
    results = inferencer(file_paths, **inference_kwargs)

    stats = InferenceStats()
    predictions = []

    logger.info("\nProcessing predictions...")
    for mmdet_pred, file_path in tqdm(zip(results["predictions"], file_paths, strict=False)):
        stats.processed_images += 1
        preds = Prediction.from_mmdet_pred(mmdet_pred, file_path, model_class_list, {})
        predictions.extend(create_prediction_df(preds, image_ids, stats))

    stats.print_summary()

    if not predictions:
        raise ValueError("No valid predictions generated")

    return pl.DataFrame(predictions)


def run_evaluation(
    pred_df: pl.DataFrame,
    annotations_df: pl.DataFrame,
    images_df: pl.DataFrame,
    model_class_list: list[str],
) -> EvalMetrics:
    """Run evaluation and return metrics."""
    # Type assertions for input parameters
    assert isinstance(pred_df, pl.DataFrame), f"pred_df must be a Polars DataFrame, got {type(pred_df)}"
    assert isinstance(annotations_df, pl.DataFrame), f"annotations_df must be a Polars DataFrame, got {type(annotations_df)}"
    assert isinstance(images_df, pl.DataFrame), f"images_df must be a Polars DataFrame, got {type(images_df)}"
    assert isinstance(model_class_list, list), f"model_class_list must be a list, got {type(model_class_list)}"
    assert all(isinstance(cls, str) for cls in model_class_list), "All elements in model_class_list must be strings"

    # Verify required columns in DataFrames
    required_pred_columns = {"score", "bbox"}
    assert all(col in pred_df.columns for col in required_pred_columns), f"Missing required columns in pred_df. Required: {required_pred_columns}"

    required_annotations_columns = {"box_x1", "box_y1", "box_width", "box_height", "class_name"}
    assert all(col in annotations_df.columns for col in required_annotations_columns), (
        f"Missing required columns in annotations_df. Required: {required_annotations_columns}"
    )

    evaluator = Evaluator(
        predictions_df=pred_df,
        annotations_df=annotations_df,
        images_df=images_df,
        model_class_list=model_class_list,
    )

    metrics_df = evaluator.calculate_metrics_at_best_thresholds(iou_threshold=0.5, beta=1)

    # Verify metrics DataFrame has expected columns
    # Add a 'name' column that's a duplicate of 'class_name'
    metrics_df = metrics_df.with_columns(pl.col("class_name").alias("name"))
    expected_metrics_columns = {"name", "precision", "recall", "best_fbeta"}
    assert all(col in metrics_df.columns for col in expected_metrics_columns), (
        f"Missing required columns in metrics_df. Required: {expected_metrics_columns}"
    )

    # Create EvalMetrics object with the class metrics
    result = EvalMetrics(class_metrics=metrics_df)

    # Final type check
    assert isinstance(result, EvalMetrics), f"Expected EvalMetrics return type, got {type(result)}"
    assert isinstance(result.class_metrics, pl.DataFrame), f"Expected class_metrics to be a DataFrame, got {type(result.class_metrics)}"

    return result


def run_model_inference(
    trained_model: TrainedModel,
    weights_path: str | None,
    config: dict,
    data_root: str,
    model_class_list: list[str],
) -> tuple[EvalMetrics, dict]:
    """Run inference for a single model and return metrics and example images."""
    logger.info(f"\nInitializing inference for {trained_model.run_name}")
    logger.info(f"Model weights: {weights_path}")

    # Load dataset
    training_split = TrainingSplit(dataset_names=[config["dataset_name"]])  # type: ignore

    # Convert included_camera_ids to the expected type
    included_camera_ids = config.get("included_camera_ids")
    if included_camera_ids is not None:
        included_ids = [included_camera_ids] if isinstance(included_camera_ids, int) else included_camera_ids
    else:
        included_ids = []  # type: list[int]

    annotations_df, images_df = training_split.get_default_training_data(
        split="test",
        included_ids=included_ids,  # Now this is always List[int]
        include_frames_with_unknown_camera_id=config.get("include_frames_with_unknown_camera_id", True),
        skip_coco_conversion=True,
    )

    # Run inference and evaluation
    inferencer = CustomDetInferencer(model=trained_model.model_config, weights=weights_path)

    # Ensure we have DataFrames before passing to filter_and_validate_images
    assert isinstance(images_df, pl.DataFrame), f"images_df must be a DataFrame, got {type(images_df)}"
    assert isinstance(annotations_df, pl.DataFrame), f"annotations_df must be a DataFrame, got {type(annotations_df)}"

    images_df, annotations_df, valid_files = filter_and_validate_images(
        images_df,  # type: ignore
        annotations_df,  # type: ignore
        data_root,
        config.get("max_frames"),
        config.get("val_set_timestamp"),
    )

    # Create image_ids dictionary
    image_ids = {os.path.basename(row["file_name"]): row["image_id"] for row in images_df.iter_rows(named=True)}

    # Run inference
    file_paths = [os.path.join(data_root, f) for f in valid_files]
    inference_kwargs = prepare_inference_kwargs(config["batch_size"])
    results = inferencer(file_paths, **inference_kwargs)

    # Process predictions
    stats = InferenceStats()
    predictions = []
    for mmdet_pred, file_path in tqdm(zip(results["predictions"], file_paths, strict=False)):
        stats.processed_images += 1
        preds = Prediction.from_mmdet_pred(mmdet_pred, file_path, model_class_list, {})
        predictions.extend(create_prediction_df(preds, image_ids, stats))

    stats.print_summary()
    pred_df = pl.DataFrame(predictions)

    # Run evaluation with bootstrap parameters from config
    metrics = run_evaluation(
        pred_df=pred_df,
        annotations_df=annotations_df,
        images_df=images_df,
        model_class_list=model_class_list,
    )

    # Before collect_example_images
    example_images = collect_example_images(
        pred_df=pred_df,
        annotations_df=annotations_df,
        data_root=data_root,
        examples_per_category=20,
        iou_threshold=0.5,
    )

    # metrics is already an EvalMetrics object because it's returned by run_evaluation
    return metrics, example_images


def collect_example_images(
    pred_df: pl.DataFrame,
    annotations_df: pl.DataFrame,
    data_root: str,
    examples_per_category: int = 20,
    iou_threshold: float = 0.5,
) -> dict[str, list[tuple[Image.Image, str]]]:
    """Collect random example images for each category (TP, FP, FN)."""
    examples = {"True Positives": [], "False Positives": [], "False Negatives": []}
    used_annotations = set()

    # Process TPs and FPs
    collect_tp_fp_examples(pred_df, annotations_df, data_root, examples, used_annotations, examples_per_category, iou_threshold)

    # Process FNs
    collect_fn_examples(annotations_df, data_root, examples, used_annotations, examples_per_category)

    return examples


def collect_tp_fp_examples(
    pred_df: pl.DataFrame,
    annotations_df: pl.DataFrame,
    data_root: str,
    examples: dict,
    used_annotations: set,
    examples_per_category: int,
    iou_threshold: float,
):
    """Collect True Positive and False Positive examples."""
    # Shuffle prediction rows for randomization
    pred_rows = list(pred_df.iter_rows(named=True))
    random.shuffle(pred_rows)

    for pred_row in pred_rows:
        if len(examples["True Positives"]) >= examples_per_category and len(examples["False Positives"]) >= examples_per_category:
            break

        pred_box = pred_row["bbox"]
        best_iou = 0
        best_annot_idx = -1

        # Find matching annotation
        for idx, annot_row in enumerate(annotations_df.iter_rows(named=True)):
            if idx in used_annotations or annot_row["class_name"] != pred_row["class_name"]:
                continue

            x1, y1, width, height = (
                annot_row["box_x1"],
                annot_row["box_y1"],
                annot_row["box_width"],
                annot_row["box_height"],
            )
            annot_box = [x1, y1, width, height]

            iou = calculate_iou(pred_box, annot_box)
            if iou > best_iou:
                best_iou = iou
                best_annot_idx = idx

        # Process the image
        process_tp_fp_image(pred_row, data_root, examples, used_annotations, best_iou, best_annot_idx, iou_threshold, examples_per_category)


def process_tp_fp_image(pred_row, data_root, examples, used_annotations, best_iou, best_annot_idx, iou_threshold, examples_per_category):
    """Process a single image for TP/FP examples."""
    image_path = os.path.join(data_root, pred_row["file_name"])
    try:
        with Image.open(image_path).convert("RGB") as img:
            expanded_box = expand_bbox(tuple(pred_row["bbox"]), img.size)
            cropped_img = img.crop(expanded_box)

            category = "True Positives" if best_iou >= iou_threshold else "False Positives"
            if len(examples[category]) < examples_per_category:
                examples[category].append((cropped_img, pred_row["class_name"]))
                if category == "True Positives":
                    used_annotations.add(best_annot_idx)
    except Exception as e:
        logger.warning(f"Error processing image {image_path}: {e}")


def collect_fn_examples(
    annotations_df: pl.DataFrame,
    data_root: str,
    examples: dict,
    used_annotations: set,
    examples_per_category: int,
):
    """Collect False Negative examples."""
    # Shuffle annotations for randomization
    annot_rows = list(annotations_df.iter_rows(named=True))
    random.shuffle(annot_rows)

    for annot_row in annot_rows:
        if len(examples["False Negatives"]) >= examples_per_category:
            break

        if annot_row["annotation_id"] in used_annotations:
            continue

        process_fn_image(annot_row, data_root, examples, used_annotations)


def process_fn_image(annot_row, data_root, examples, used_annotations):
    """Process a single image for FN examples."""
    image_path = os.path.join(data_root, annot_row["file_name"])
    try:
        with Image.open(image_path).convert("RGB") as img:
            x1, y1, width, height = (
                annot_row["box_x1"],
                annot_row["box_y1"],
                annot_row["box_width"],
                annot_row["box_height"],
            )
            annot_box = [x1, y1, width, height]
            expanded_box = expand_bbox(tuple(annot_box), img.size)
            cropped_img = img.crop(expanded_box)
            examples["False Negatives"].append((cropped_img, annot_row["class_name"]))
            used_annotations.add(annot_row["annotation_id"])
    except Exception as e:
        logger.warning(f"Error processing image {image_path}: {e}")


def prepare_inference_kwargs(
    batch_size: int,
    use_sahi: bool = False,
    sahi_slice_height: int = 1000,
    sahi_slice_width: int = 1000,
    sahi_overlap_height_ratio: float = 0.2,
    sahi_overlap_width_ratio: float = 0.2,
) -> dict:
    """Prepare inference keyword arguments."""
    kwargs = {
        "return_vis": False,
        "batch_size": batch_size,
        "no_save_pred": True,
        "pred_score_thr": 0,
        "no_save_vis": True,
        "out_dir": "outputs",
    }

    if use_sahi:
        kwargs.update(
            {
                "sahi_slice_height": sahi_slice_height,
                "sahi_slice_width": sahi_slice_width,
                "sahi_overlap_height_ratio": sahi_overlap_height_ratio,
                "sahi_overlap_width_ratio": sahi_overlap_width_ratio,
            }
        )

    return kwargs


def load_inference_config(config_path: str) -> dict:
    """
    Load inference configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing inference configuration
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Convert None strings to actual None
    for key, value in config.items():
        if isinstance(value, str) and value.lower() == "none":
            config[key] = None

    # Convert included_camera_ids to list if specified
    if config.get("included_camera_ids"):
        if isinstance(config["included_camera_ids"], str):
            config["included_camera_ids"] = [int(x.strip()) for x in config["included_camera_ids"].split(",")]

    # Convert max_frames to int if specified
    if config.get("max_frames"):
        try:
            config["max_frames"] = int(config["max_frames"])
        except ValueError:
            logger.warning(f"Invalid max_frames value in config: {config['max_frames']}")
            config["max_frames"] = None

    return config


def filter_and_validate_images(
    images_df: pl.DataFrame,
    annotations_df: pl.DataFrame,
    data_root: str,
    max_frames: int | None = None,
    val_set_timestamp: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    """Filter images by frame URI, validate files, and optionally limit frame count."""
    # Ensure we have the correct types
    assert isinstance(images_df, pl.DataFrame), f"images_df must be a DataFrame, got {type(images_df)}"
    assert isinstance(annotations_df, pl.DataFrame), f"annotations_df must be a DataFrame, got {type(annotations_df)}"
    # First deduplicate by frame URI
    images_df = images_df.group_by("frame_uri").first()
    logger.info(f"Initial unique frames: {len(images_df)}")

    # Apply timestamp filter if specified
    if val_set_timestamp:
        images_df = images_df.filter(pl.col("update_timestamp") < val_set_timestamp)
        logger.info(f"Frames after timestamp filter: {len(images_df)}")

    # Apply max_frames limit if specified
    if max_frames and max_frames > 0 and len(images_df) > max_frames:
        sampled_uris = pl.Series(images_df["frame_uri"]).sample(max_frames, seed=42)
        images_df = images_df.filter(pl.col("frame_uri").is_in(sampled_uris))
        logger.info(f"Sampled down to {max_frames} frames")

    # Validate file existence
    valid_files = [
        f for f in images_df["file_name"] if os.path.exists(os.path.join(data_root, f)) and os.path.getsize(os.path.join(data_root, f)) > 0
    ]
    logger.info(f"Found {len(valid_files)} valid files out of {len(images_df)} total files")

    # Filter to valid files and get final frame URIs
    images_df = images_df.filter(pl.col("file_name").is_in(valid_files))
    valid_frame_uris = images_df["frame_uri"].unique()

    # Filter annotations
    annotations_df = annotations_df.filter(pl.col("frame_uri").is_in(valid_frame_uris))
    logger.info(f"Final dataset: {len(images_df)} images, {len(annotations_df)} annotations")

    return images_df, annotations_df, valid_files


def run_pipeline_inference(
    inference_config_name: str,
    dataset_name: str,
    neptune_run_id: str,
    neptune_project: str,
    batch_size: int = 10,
) -> EvalMetrics:
    model = TrainedModel.from_neptune(neptune_project=neptune_project, neptune_run_id=neptune_run_id)

    # Load config and override specific parameters
    config = get_inference_config_by_name(inference_config_name)
    if config is None:
        raise ValueError(f"Inference config '{inference_config_name}' not found.")
    logger.info(config)
    config["dataset_name"] = dataset_name
    config["batch_size"] = batch_size
    config["compare_with_deployed"] = False

    return _run_inference(config, pipeline_model=model)


def main(
    config_name: str,
    pipeline_model=None,
) -> EvalMetrics:
    config_dict = get_inference_config_by_name(config_name)
    if config_dict is None:
        raise ValueError(f"Inference config '{config_name}' not found.")

    logger.info(config_name)
    return _run_inference(config_dict, pipeline_model)


def _run_inference(
    config_dict: dict,
    pipeline_model=None,
) -> EvalMetrics:
    """Run model inference and evaluation on validation set."""
    # Setup and configuration
    data_root = env.image_directory

    # Get model information
    trained_model, model_info = setup_model(config_dict, pipeline_model)

    # Run inference and evaluation
    metrics, example_images = run_model_inference(
        trained_model,
        trained_model.get_model_weights(),
        config_dict,
        data_root,
        trained_model.get_class_list(),
    )

    # Create and configure report
    report = create_model_report(trained_model, metrics, example_images)

    # Handle deployed model comparison if needed
    deployed_model = handle_deployed_model_comparison(config_dict, data_root, report, metrics)

    # Save and share report
    save_and_share_report(report, config_dict, model_info, trained_model, deployed_model)

    return metrics


def setup_model(config_dict, pipeline_model):
    """Set up the model for inference."""
    model_info = config_dict["model_names"][0]

    if pipeline_model is None:
        trained_model = TrainedModel(
            run_name=model_info["run_name"],
            step_number=model_info["step_number"],
            model_name=config_dict["dataset_name"],
        )
    else:
        trained_model = pipeline_model
        model_info = {
            "run_name": trained_model.run_name,
            "step_number": trained_model.step_number,
        }

    return trained_model, model_info


def create_model_report(trained_model, metrics, example_images):
    """Create the model evaluation report."""
    report = ModelEvaluationReport()
    report.add_title_page(
        title="Model Evaluation Report",
        model_name=f"{trained_model.run_name} (iteration {trained_model.step_number})",
    )
    report.add_evaluation_metrics(
        {"class_metrics": metrics.class_metrics, "group_metrics": metrics.group_metrics},
        trained_model.get_class_list(),
    )
    report.add_example_images(example_images)

    return report


def handle_deployed_model_comparison(config_dict, data_root, report, current_metrics):
    """Handle comparison with deployed model if configured."""
    deployed_model = None

    if config_dict.get("deployed_model"):
        deployed_model_info = config_dict["deployed_model"][0]
        deployed_model = TrainedModel(
            run_name=deployed_model_info["run_name"],
            step_number=deployed_model_info["step_number"],
        )

        deployed_metrics, _ = run_model_inference(
            deployed_model,
            deployed_model.get_model_weights(),
            config_dict,
            data_root,
            deployed_model.get_class_list(),
        )

        # Debug print
        print("Deployed model metrics schema:")
        print(deployed_metrics.class_metrics.schema)

        # Create comparison DataFrames
        comparison_df, group_comparison_df = create_comparison_dataframes(current_metrics, deployed_metrics)

        # Add comparison to report
        report.add_comparison_section(comparison_df, group_comparison_df)

    return deployed_model


def create_comparison_dataframes(current_metrics, deployed_metrics):
    """Create comparison DataFrames between current and deployed models."""
    # Create comparison DataFrame for classes
    comparison_data = []
    all_classes = set(current_metrics.class_metrics["name"].to_list()) | set(deployed_metrics.class_metrics["name"].to_list())

    metrics_to_compare = [
        ("best_fbeta", "fbeta", 0.0),
        ("precision", "precision", 0.0),
        ("recall", "recall", 0.0),
        ("tp", "tp", 0),
        ("fp", "fp", 0),
        ("fn", "fn", 0),
    ]

    for class_name in sorted(all_classes):
        row_data = {"name": class_name}
        for metric_name, output_name, default in metrics_to_compare:
            old_val = get_metric_value(deployed_metrics.class_metrics, class_name, metric_name, default)
            new_val = get_metric_value(current_metrics.class_metrics, class_name, metric_name, default)
            row_data[f"{output_name}_old"] = old_val
            row_data[f"{output_name}_new"] = new_val
        comparison_data.append(row_data)

    # Create DataFrame with explicit schema
    schema = {
        "name": pl.Utf8,
        "fbeta_old": pl.Float64,
        "fbeta_new": pl.Float64,
        "precision_old": pl.Float64,
        "precision_new": pl.Float64,
        "recall_old": pl.Float64,
        "recall_new": pl.Float64,
        "tp_old": pl.Int64,
        "tp_new": pl.Int64,
        "fp_old": pl.Int64,
        "fp_new": pl.Int64,
        "fn_old": pl.Int64,
        "fn_new": pl.Int64,
    }

    comparison_df = pl.DataFrame(comparison_data, schema=schema)

    # Create comparison DataFrame for groups if available
    group_comparison_df = None
    if (
        hasattr(current_metrics, "group_metrics")
        and hasattr(deployed_metrics, "group_metrics")
        and current_metrics.group_metrics is not None
        and deployed_metrics.group_metrics is not None
    ):
        group_comparison_df = create_group_comparison_df(current_metrics, deployed_metrics, metrics_to_compare, schema)

    return comparison_df, group_comparison_df


def create_group_comparison_df(current_metrics, deployed_metrics, metrics_to_compare, schema):
    """Create comparison DataFrame for group metrics."""
    group_comparison_data = []
    all_groups = set(current_metrics.group_metrics["name"].to_list()) | set(deployed_metrics.group_metrics["name"].to_list())

    for group_name in sorted(all_groups):
        row_data = {"name": group_name}
        for metric_name, output_name, default in metrics_to_compare:
            old_val = get_metric_value(deployed_metrics.group_metrics, group_name, metric_name, default)
            new_val = get_metric_value(current_metrics.group_metrics, group_name, metric_name, default)
            row_data[f"{output_name}_old"] = old_val
            row_data[f"{output_name}_new"] = new_val
        group_comparison_data.append(row_data)

    return pl.DataFrame(group_comparison_data, schema=schema)


def get_metric_value(df, class_name, metric_name, default_value=0.0):
    """Get metric value with default if class not found."""
    rows = df.filter(pl.col("name") == class_name)
    if len(rows) == 0:
        return default_value
    # Extract scalar value from Series
    value = rows[0][metric_name]
    if isinstance(value, pl.Series):
        value = value.item()  # Convert Series to scalar
    return value


def save_and_share_report(report, config_dict, model_info, trained_model, deployed_model):
    """Save the report and optionally share it via Slack."""
    report_path = "evaluation_report.pdf"
    report.save(report_path)
    logger.info(f"Evaluation report generated: {report_path}")

    if config_dict.get("send_to_slack", True):
        deployed_models = [deployed_model] if deployed_model is not None else None
        send_file_to_slack(
            image_path=report_path,
            message=create_slack_message(
                model_info=[model_info],
                trained_model=trained_model,
                dataset_name=config_dict["dataset_name"],
                deployed_model=deployed_models,
                val_set_timestamp=config_dict.get("val_set_timestamp"),
            ),
        )

    return report_path


def create_slack_message(
    model_info: list[dict],
    trained_model: TrainedModel,
    dataset_name: str,
    deployed_model: list[TrainedModel] | None = None,
    val_set_timestamp: str | None = None,
) -> str:
    """Create message for Slack notification.

    Args:
        model_info: List of dictionaries containing model information
        trained_model: Current model instance
        dataset_name: Name of the dataset
        deployed_model: Optional list of deployed models for comparison
        val_set_timestamp: Optional timestamp for validation set cutoff

    Returns:
        Formatted message string
    """
    timestamp = val_set_timestamp if val_set_timestamp else datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    message = ["Model Comparison Report:"]

    # Add new models section
    message.append("New Models:")
    for model in model_info:
        message.extend([f"  Run name: {model['run_name']}", f"  Iteration: {model['step_number']}"])
        if model != model_info[-1]:  # Add separator between models
            message.append("  ---")

    # Add deployed models section if present
    if deployed_model:
        message.append("Deployed Models:")
        for model in deployed_model:
            message.extend([f"  Run name: {model.run_name}", f"  Iteration: {model.step_number}"])
            if model != deployed_model[-1]:  # Add separator between models
                message.append("  ---")

    # Add dataset information
    message.extend([f"Dataset: {dataset_name}", f"Validation Set: {timestamp}"])

    return "\n".join(message)


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First box in format [x, y, width, height]
        box2: Second box in format [x, y, width, height]

    Returns:
        IoU value between 0 and 1
    """
    if any(dim < 0 for dim in box1[2:] + box2[2:]):
        raise ValueError("Box dimensions must be non-negative")
    # Convert from [x, y, width, height] to [x1, y1, x2, y2]
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]

    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]

    # Calculate intersection coordinates
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection

    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0

    return iou


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_show_locals=False)
    app.command()(main)
    app()
