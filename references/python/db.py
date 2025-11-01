import os
import re
from datetime import datetime

import polars as pl
import psycopg
from loguru import logger


def extract_timestamp_from_frame_uri(frame_uri: str) -> datetime | None:
    """
    Extract the timestamp from a frame URI.

    Example frame_uri: gs://ingestion_prod/frames/120d-2025-05-20T14:27:36.390962496Z-138.jpg
    or: gs://ingestion_prod/frames/120d-2025-07-23T11:52:09.689-04:00-138.jpg

    Returns:
        datetime object representing the timestamp in the URI (naive datetime, no timezone info)
    """
    # Extract the ISO 8601 timestamp using regex - handle both Z and timezone offset formats
    match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+(?:Z|[+-]\d{2}:\d{2}))", frame_uri)
    if match:
        timestamp_str = match.group(1)
        # Handle Z format (UTC)
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str.rstrip("Z")
        # Truncate to 6 digits for microseconds if longer
        if "." in timestamp_str:
            main_part, microseconds = timestamp_str.split(".")
            if len(microseconds) > 6:
                microseconds = microseconds[:6]
            timestamp_str = f"{main_part}.{microseconds}"

        # Parse the datetime and convert to naive datetime (no timezone)
        dt = datetime.fromisoformat(timestamp_str)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    return None


def get_connection():
    return psycopg.connect(os.environ["PG_DATABASE_URL"])


def save_df_to_monitoring_table(metrics_df: pl.DataFrame, frame_uri: str, pipeline_name: str, git_hash: str):
    """
    Saves a polars dataframe the table machine_learning.model_monitoring
    metrics_df: Dataframe containing the metrics to save. The dataframe should have the following columns:
        - name: class name (str)
        - tp: number of true positives (int)
        - fp: number of false positives (int)
        - fn: number of false negatives (int)
        - precision: precision (float)
        - recall: recall (float)
        - fbeta: f1 score (float)
    pipeline_name: name of the pipeline that generated the predictions/metrics (str)
    git_hash: Git hash of the pipeline
    cvat_task_job_id: ID of the CVAT task/job that generated the metrics
    """
    with get_connection() as conn:
        # Set autocommit to False to handle transactions explicitly
        conn.autocommit = False

        # Create a cursor
        with conn.cursor() as cur:
            created_at_timestamp = extract_timestamp_from_frame_uri(frame_uri)
            if created_at_timestamp is None:
                raise ValueError(f"Could not extract timestamp from frame URI: {frame_uri}")
            for row in metrics_df.iter_rows(named=True):
                query = """
                INSERT INTO machine_learning.pipeline_monitoring (pipeline_name, frame_uri, git_hash, class_name, true_positives,
                false_positives, false_negatives, precision, recall, f1_score, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                params = (
                    pipeline_name,
                    frame_uri,
                    git_hash,
                    row["name"],
                    row["tp"],
                    row["fp"],
                    row["fn"],
                    row["precision"],
                    row["recall"],
                    row["fbeta"],
                    created_at_timestamp,
                )

                try:
                    # Execute the query with parameters
                    cur.execute(query, params)
                except Exception as e:
                    # If we get unique constraint violation, it's likely we're inserting a duplicate
                    # Just log and continue
                    logger.warning(f"Error inserting threshold data: {e}")
                    # Rollback this specific insert but continue with others
                    conn.rollback()
                    raise e
                else:
                    # Commit on success
                    conn.commit()
