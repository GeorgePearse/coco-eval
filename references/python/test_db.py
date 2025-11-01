import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from evaluation.db import extract_timestamp_from_frame_uri, get_connection, save_df_to_monitoring_table


def test_extract_timestamp_from_frame_uri_valid():
    """Test extracting timestamp from a valid frame URI."""
    frame_uri = "gs://ingestion_prod/frames/120d-2025-05-20T14:27:36.390962496Z-138.jpg"
    result = extract_timestamp_from_frame_uri(frame_uri)

    assert isinstance(result, datetime)
    assert result.year == 2025
    assert result.month == 5
    assert result.day == 20
    assert result.hour == 14
    assert result.minute == 27
    assert result.second == 36
    assert result.microsecond == 390962  # Truncated to 6 digits


def test_extract_timestamp_from_frame_uri_different_format():
    """Test extracting timestamp with different microsecond precision."""
    frame_uri = "gs://bucket/frames/cam1-2024-12-31T23:59:59.123Z-001.jpg"
    result = extract_timestamp_from_frame_uri(frame_uri)

    assert isinstance(result, datetime)
    assert result.year == 2024
    assert result.month == 12
    assert result.day == 31
    assert result.hour == 23
    assert result.minute == 59
    assert result.second == 59
    assert result.microsecond == 123000


def test_extract_timestamp_from_frame_uri_no_microseconds():
    """Test extracting timestamp without microseconds returns None."""
    frame_uri = "gs://bucket/frames/cam2-2023-01-15T08:30:45Z-999.jpg"
    result = extract_timestamp_from_frame_uri(frame_uri)

    # The regex requires microseconds, so this should return None
    assert result is None


def test_extract_timestamp_from_frame_uri_invalid():
    """Test extracting timestamp from invalid URIs."""
    invalid_uris = [
        "gs://bucket/frames/no-timestamp.jpg",
        "invalid-uri",
        "",
        "gs://bucket/frames/cam1-invalid-timestamp-001.jpg",
        "gs://bucket/frames/cam1-2023-13-45T25:70:90Z-001.jpg",  # Invalid date/time
    ]

    for uri in invalid_uris:
        result = extract_timestamp_from_frame_uri(uri)
        assert result is None


def test_get_connection():
    """Test get_connection function with mocked environment variable."""
    with patch.dict(os.environ, {"PG_DATABASE_URL": "postgresql://user:pass@localhost/db"}):
        with patch("psycopg.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            result = get_connection()

            mock_connect.assert_called_once_with("postgresql://user:pass@localhost/db")
            assert result == mock_conn


def test_get_connection_missing_env_var():
    """Test get_connection when environment variable is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError):
            get_connection()


def test_save_df_to_monitoring_table_success():
    """Test successful save to monitoring table."""
    metrics_df = pl.DataFrame(
        {
            "name": ["cat", "dog"],
            "tp": [10, 5],
            "fp": [2, 1],
            "fn": [3, 4],
            "precision": [0.833, 0.833],
            "recall": [0.769, 0.556],
            "fbeta": [0.800, 0.667],
        }
    )

    frame_uri = "gs://bucket/frames/test-2025-06-01T12:00:00.000000Z-001.jpg"
    pipeline_name = "test_pipeline"
    git_hash = "abc123def"

    with patch("evaluation.db.get_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None

        save_df_to_monitoring_table(metrics_df, frame_uri, pipeline_name, git_hash)

        # Verify cursor.execute was called twice (once for each row)
        assert mock_cursor.execute.call_count == 2

        # Verify commit was called twice
        assert mock_conn.commit.call_count == 2

        # Check the parameters of the first call
        first_call_args = mock_cursor.execute.call_args_list[0]
        query = first_call_args[0][0]
        params = first_call_args[0][1]

        assert "INSERT INTO machine_learning.pipeline_monitoring" in query
        assert params[0] == pipeline_name
        assert params[1] == frame_uri
        assert params[2] == git_hash
        assert params[3] == "cat"
        assert params[4] == 10  # tp
        assert params[5] == 2  # fp
        assert params[6] == 3  # fn


def test_save_df_to_monitoring_table_invalid_frame_uri():
    """Test save_df_to_monitoring_table with invalid frame URI."""
    metrics_df = pl.DataFrame({"name": ["cat"], "tp": [10], "fp": [2], "fn": [3], "precision": [0.833], "recall": [0.769], "fbeta": [0.800]})

    frame_uri = "invalid-frame-uri"
    pipeline_name = "test_pipeline"
    git_hash = "abc123def"

    with patch("evaluation.db.get_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None

        with pytest.raises(ValueError, match="Could not extract timestamp from frame URI"):
            save_df_to_monitoring_table(metrics_df, frame_uri, pipeline_name, git_hash)


def test_save_df_to_monitoring_table_database_error():
    """Test handling of database errors during insert."""
    metrics_df = pl.DataFrame({"name": ["cat"], "tp": [10], "fp": [2], "fn": [3], "precision": [0.833], "recall": [0.769], "fbeta": [0.800]})

    frame_uri = "gs://bucket/frames/test-2025-06-01T12:00:00.000000Z-001.jpg"
    pipeline_name = "test_pipeline"
    git_hash = "abc123def"

    with patch("evaluation.db.get_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Database error")
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None

        with pytest.raises(Exception, match="Database error"):
            save_df_to_monitoring_table(metrics_df, frame_uri, pipeline_name, git_hash)

        # Verify rollback was called
        mock_conn.rollback.assert_called_once()


def test_save_df_to_monitoring_table_empty_dataframe():
    """Test save_df_to_monitoring_table with empty DataFrame."""
    metrics_df = pl.DataFrame({"name": [], "tp": [], "fp": [], "fn": [], "precision": [], "recall": [], "fbeta": []})

    frame_uri = "gs://bucket/frames/test-2025-06-01T12:00:00.000000Z-001.jpg"
    pipeline_name = "test_pipeline"
    git_hash = "abc123def"

    with patch("evaluation.db.get_connection") as mock_get_conn:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None

        save_df_to_monitoring_table(metrics_df, frame_uri, pipeline_name, git_hash)

        # Verify cursor.execute was never called (no rows to insert)
        mock_cursor.execute.assert_not_called()
