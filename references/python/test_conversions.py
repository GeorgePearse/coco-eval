import polars as pl

from evaluation.conversions import convert_offline_inference_preds_to_predictions_df


def test_convert_offline_inference_preds_basic():
    """Test basic conversion with valid predictions."""
    items = pl.DataFrame(
        {
            "frame_uri": ["gs://bucket/frames/test1.jpg"],
            "width": [640],
            "height": [480],
            "detections.x": [[0.1]],
            "detections.y": [[0.2]],
            "detections.w": [[0.5]],
            "detections.h": [[0.6]],
            "detections.filtered_by": [[""]],  # Should be a list of lists
            "detections.score": [[0.95]],
            "detections.segmentation_points": [[[{"x": 0.1, "y": 0.2}, {"x": 0.5, "y": 0.6}]]],
            "inference_category_name": [["cat"]],
            "pipeline_name": ["test_pipeline"],
            "git_hash": ["abc123"],
        }
    )

    filename_to_id = {"test1.jpg": 1}

    result = convert_offline_inference_preds_to_predictions_df(items, filename_to_id)

    assert len(result) == 1
    assert result["file_name"][0] == "test1.jpg"
    assert result["image_id"][0] == 1
    assert result["class_name"][0] == "cat"
    assert result["score"][0] == 0.95
    bbox = result["bbox"][0]
    # Check width calculation: (0.5 - 0.1) * 640 = 0.4 * 640 = 256
    # Check height calculation: (0.6 - 0.2) * 480 = 0.4 * 480 = 192
    import numpy as np

    assert np.allclose(bbox.to_list(), [64.0, 96.0, 256.0, 192.0])
    # Check segmentation points
    seg_points = result["segmentation_points"][0]
    assert len(seg_points) == 2
    assert seg_points[0].to_list() == [0.1, 0.2]
    assert seg_points[1].to_list() == [0.5, 0.6]
    assert result["pipeline_name"][0] == "test_pipeline"
    assert result["git_hash"][0] == "abc123"


def test_convert_offline_inference_preds_multiple_detections():
    """Test conversion with multiple detections in one image."""
    items = pl.DataFrame(
        {
            "frame_uri": ["gs://bucket/frames/test2.jpg"],
            "width": [800],
            "height": [600],
            "detections.x": [[0.1, 0.6]],
            "detections.y": [[0.1, 0.5]],
            "detections.w": [[0.3, 0.9]],
            "detections.h": [[0.4, 0.8]],
            "detections.filtered_by": [["", ""]],
            "detections.score": [[0.8, 0.9]],
            "detections.segmentation_points": [[[{"x": 0.1, "y": 0.1}], [{"x": 0.6, "y": 0.5}]]],
            "inference_category_name": [["dog", "cat"]],
            "pipeline_name": ["test_pipeline"],
            "git_hash": ["def456"],
        }
    )

    filename_to_id = {"test2.jpg": 2}

    result = convert_offline_inference_preds_to_predictions_df(items, filename_to_id)

    assert len(result) == 2
    assert result["class_name"][0] == "dog"
    assert result["class_name"][1] == "cat"
    assert result["score"][0] == 0.8
    assert result["score"][1] == 0.9


def test_convert_offline_inference_preds_filtered_detections():
    """Test that filtered detections are excluded."""
    items = pl.DataFrame(
        {
            "frame_uri": ["gs://bucket/frames/test3.jpg"],
            "width": [640],
            "height": [480],
            "detections.x": [[0.1, 0.5]],
            "detections.y": [[0.1, 0.5]],
            "detections.w": [[0.3, 0.7]],
            "detections.h": [[0.3, 0.8]],
            "detections.filtered_by": [["", "nms"]],
            "detections.score": [[0.8, 0.7]],
            "detections.segmentation_points": [[[{"x": 0.1, "y": 0.1}], [{"x": 0.5, "y": 0.5}]]],
            "inference_category_name": [["cat", "cat"]],
            "pipeline_name": ["test_pipeline"],
            "git_hash": ["ghi789"],
        }
    )

    filename_to_id = {"test3.jpg": 3}

    result = convert_offline_inference_preds_to_predictions_df(items, filename_to_id)

    assert len(result) == 1  # Only the unfiltered detection
    assert result["score"][0] == 0.8


def test_convert_offline_inference_preds_replaced_by_classifier():
    """Test that detections filtered by 'replaced_by_classifier' are included."""
    items = pl.DataFrame(
        {
            "frame_uri": ["gs://bucket/frames/test4.jpg"],
            "width": [640],
            "height": [480],
            "detections.x": [[0.1, 0.5]],
            "detections.y": [[0.1, 0.5]],
            "detections.w": [[0.3, 0.7]],
            "detections.h": [[0.3, 0.8]],
            "detections.filtered_by": [["replaced_by_classifier", ""]],
            "detections.score": [[0.8, 0.7]],
            "detections.segmentation_points": [[[{"x": 0.1, "y": 0.1}], [{"x": 0.5, "y": 0.5}]]],
            "inference_category_name": [["cat", "dog"]],
            "pipeline_name": ["test_pipeline"],
            "git_hash": ["jkl012"],
        }
    )

    filename_to_id = {"test4.jpg": 4}

    result = convert_offline_inference_preds_to_predictions_df(items, filename_to_id)

    assert len(result) == 2  # Both detections should be included


def test_convert_offline_inference_preds_zero_dimensions():
    """Test that detections with zero width or height are excluded."""
    items = pl.DataFrame(
        {
            "frame_uri": ["gs://bucket/frames/test5.jpg"],
            "width": [640],
            "height": [480],
            "detections.x": [[0.1, 0.5, 0.7]],
            "detections.y": [[0.1, 0.5, 0.3]],
            "detections.w": [[0.1, 0.5, 1.0]],  # w is x2, so width = (0.1-0.1)*640 = 0
            "detections.h": [[0.4, 0.5, 0.8]],  # h is y2, so height = (0.5-0.5)*480 = 0
            "detections.filtered_by": [["", "", ""]],
            "detections.score": [[0.8, 0.7, 0.9]],
            "detections.segmentation_points": [[[{"x": 0.1, "y": 0.1}], [{"x": 0.5, "y": 0.5}], [{"x": 0.7, "y": 0.3}]]],
            "inference_category_name": [["cat", "dog", "bird"]],
            "pipeline_name": ["test_pipeline"],
            "git_hash": ["mno345"],
        }
    )

    filename_to_id = {"test5.jpg": 5}

    result = convert_offline_inference_preds_to_predictions_df(items, filename_to_id)

    assert len(result) == 1  # Only the third detection has non-zero dimensions
    assert result["class_name"][0] == "bird"


def test_convert_offline_inference_preds_empty_input():
    """Test conversion with empty input."""
    items = pl.DataFrame(
        {
            "frame_uri": [],
            "width": [],
            "height": [],
            "detections.x": [],
            "detections.y": [],
            "detections.w": [],
            "detections.h": [],
            "detections.filtered_by": [],
            "detections.score": [],
            "detections.segmentation_points": [],
            "inference_category_name": [],
            "pipeline_name": [],
            "git_hash": [],
        }
    )

    filename_to_id = {}

    result = convert_offline_inference_preds_to_predictions_df(items, filename_to_id)

    assert len(result) == 0


def test_convert_offline_inference_preds_no_detections():
    """Test conversion with image but no detections."""
    items = pl.DataFrame(
        {
            "frame_uri": ["gs://bucket/frames/test6.jpg"],
            "width": [640],
            "height": [480],
            "detections.x": [[]],
            "detections.y": [[]],
            "detections.w": [[]],
            "detections.h": [[]],
            "detections.filtered_by": [[]],
            "detections.score": [[]],
            "detections.segmentation_points": [[]],
            "inference_category_name": [[]],
            "pipeline_name": ["test_pipeline"],
            "git_hash": ["pqr678"],
        }
    )

    filename_to_id = {"test6.jpg": 6}

    result = convert_offline_inference_preds_to_predictions_df(items, filename_to_id)

    assert len(result) == 0
