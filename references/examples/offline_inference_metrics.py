import uuid

import pandas as pd
import polars as pl

from evaluation.eval_metrics import Evaluator
from offline_inference.clickhouse import get_clickhouse_client, save_metrics_to_clickhouse


def compute_and_store_metrics(offline_inference_id: str, ml_dataset_name: str):
    # Pull predictions from default.offline_frame_events table
    # Only select columns we actually need to reduce data transfer
    clickhouse_client = get_clickhouse_client()
    result_preds = clickhouse_client.query_df(
        """
        select frame_uri,
               width,
               height,
               pipeline_name,
               git_hash,
               detections.x,
               detections.y,
               detections.w,
               detections.h,
               detections.score,
               detections.filtered_by,
               detections.segmentation_points,
               arrayMap(
                       id -> dictGetOrNull(
                               'default.inference_categories_dict',
                               'name',
                               id
                             ),
                       detections.inference_category_id
               ) AS inference_category_name
        from default.offline_frame_events
        where JSONExtractString(extra_data, 'offline_inference_id') = %(offline_inference_id)s
        """,
        parameters={"offline_inference_id": offline_inference_id},
    )

    # Coerce UUIDs -> strings before converting to Polars
    result_preds = _coerce_uuid_cols_to_str(result_preds)

    # Convert predictions to Polars DataFrame
    df_preds = pl.DataFrame(result_preds)

    # Pull annotations from machine_learning.labelled_datasets
    # Only select columns needed for evaluation to reduce data transfer
    result_gt = clickhouse_client.query_df(
        "select * except (segmentation) from psql.labelled_datasets final where dataset_name = %(ml_dataset_name)s and training_split = 'test'",
        parameters={"ml_dataset_name": ml_dataset_name},
    )

    # Coerce UUIDs -> strings before converting to Polars
    result_gt = _coerce_uuid_cols_to_str(result_gt)

    # Convert ground truth to Polars DataFrame
    df_gt = pl.DataFrame(result_gt)
    df_gt = df_gt.with_columns(pl.col("frame_uri").str.split("/").list.last().alias("file_name"))
    df_gt = df_gt.filter(pl.col("frame_uri").is_in(df_preds["frame_uri"]))

    # Create images df for evaluator
    frame_uris = df_preds["frame_uri"].unique().to_list()
    images_df = pl.DataFrame(
        {
            "image_id": pl.Series(range(1, len(frame_uris) + 1), dtype=pl.Int32),
            "file_name": pl.Series(frame_uris).str.split("/").list.last(),
        }
    )

    filename_to_id = dict(zip(images_df["file_name"], images_df["image_id"], strict=False))
    df_preds = convert_offline_inference_preds_to_predictions_df(df_preds, filename_to_id)

    # Create polarseval object and calculate f1 score (calculate_fbeta)
    evaluator = Evaluator(
        annotations_df=df_gt,
        images_df=images_df,
        predictions_df=df_preds,
        model_class_list=df_gt["class_name"].unique().to_list(),
    )
    metrics_df = evaluator.calculate_metrics_at_best_thresholds(beta=1)

    save_metrics_to_clickhouse(metrics_df, offline_inference_id)


def convert_offline_inference_preds_to_predictions_df(
    items: pl.DataFrame,
    filename_to_id: dict[str, int],
):
    """
    Convert inference response to a Polars DataFrame for evaluation.

    Args:
        items: DataFrame with nested detection arrays
        filename_to_id: Mapping from filenames to image IDs

    Returns:
        Polars DataFrame with one row per detection
    """
    if items.is_empty():
        return pl.DataFrame(
            schema={
                "file_name": pl.Utf8,
                "image_id": pl.Int32,
                "class_name": pl.Utf8,
                "score": pl.Float64,
                "bbox": pl.List(pl.Float64),
                "segmentation_points": pl.List(pl.List(pl.Float64)),
                "pipeline_name": pl.Utf8,
                "git_hash": pl.Utf8,
            }
        )

    # Create filename to image_id DataFrame for joining
    filename_id_df = pl.DataFrame(
        {
            "file_name": list(filename_to_id.keys()),
            "image_id": list(filename_to_id.values()),
        }
    )

    # Explode all detection arrays simultaneously
    df = items.select(
        [
            pl.col("frame_uri"),
            pl.col("width"),
            pl.col("height"),
            pl.col("pipeline_name"),
            pl.col("git_hash"),
            pl.col("detections.x").alias("x"),
            pl.col("detections.y").alias("y"),
            pl.col("detections.w").alias("w"),
            pl.col("detections.h").alias("h"),
            pl.col("detections.score").alias("score"),
            pl.col("detections.filtered_by").alias("filtered_by"),
            pl.col("detections.segmentation_points").alias("seg_points"),
            pl.col("inference_category_name"),
        ]
    ).explode(["x", "y", "w", "h", "score", "filtered_by", "seg_points", "inference_category_name"])

    # Filter out detections that were filtered by NMS or other processes
    # Keep only empty strings or 'replaced_by_classifier'
    df = df.filter((pl.col("filtered_by") == "") | (pl.col("filtered_by") == "replaced_by_classifier"))

    # Convert normalized coordinates to absolute pixels
    df = df.with_columns(
        [
            # Convert normalized coords to absolute pixels
            (pl.col("x") * pl.col("width")).alias("x1_abs"),
            (pl.col("y") * pl.col("height")).alias("y1_abs"),
            ((pl.col("w") - pl.col("x")) * pl.col("width")).alias("width_abs"),
            ((pl.col("h") - pl.col("y")) * pl.col("height")).alias("height_abs"),
            pl.col("frame_uri").str.split("/").list.last().alias("file_name"),
        ]
    )

    # Filter out invalid boxes (zero or negative width/height)
    df = df.filter((pl.col("width_abs") > 0) & (pl.col("height_abs") > 0))

    # Create bbox as list column
    df = df.with_columns(
        [
            pl.concat_list(
                [
                    pl.col("x1_abs"),
                    pl.col("y1_abs"),
                    pl.col("width_abs"),
                    pl.col("height_abs"),
                ]
            ).alias("bbox"),
        ]
    )

    # Join with filename_id_df to add image_id
    df = df.join(filename_id_df, on="file_name", how="left")

    # Select and rename final columns
    result = df.select(
        [
            pl.col("file_name"),
            pl.col("image_id"),
            pl.col("inference_category_name").alias("class_name"),
            pl.col("score"),
            pl.col("bbox"),
            pl.col("seg_points").alias("segmentation_points"),
            pl.col("pipeline_name"),
            pl.col("git_hash"),
        ]
    )

    return result


def _coerce_uuid_cols_to_str(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        s = df[col]
        # Only touch columns that actually contain UUIDs
        if s.map(lambda v: isinstance(v, uuid.UUID)).any():
            df[col] = s.map(lambda v: str(v) if isinstance(v, uuid.UUID) else (None if pd.isna(v) else v))
    return df
