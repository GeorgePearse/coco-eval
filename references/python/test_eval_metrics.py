import numpy as np
import polars as pl

from evaluation.eval_metrics import Evaluator, validate_bbox


# Test validate_bbox function
def test_validate_bbox_valid():
    # Test valid bounding box
    assert validate_bbox([10, 20, 30, 40])


def test_validate_bbox_invalid_length():
    # Test invalid length
    assert not validate_bbox([10, 20, 30])
    assert not validate_bbox([10, 20, 30, 40, 50])


def test_validate_bbox_negative_dimensions():
    # Test negative or zero dimensions
    assert not validate_bbox([10, 20, -30, 40])
    assert not validate_bbox([10, 20, 0, 40])
    assert not validate_bbox([10, 20, 30, 0])


def test_validate_bbox_invalid_types():
    # Test invalid types
    assert validate_bbox([10, 20, "30", 40])  # String numbers are converted to float
    assert not validate_bbox([10, None, 30, 40])


def test_validate_bbox_nan_inf():
    # Test NaN or infinite values
    assert not validate_bbox([10, 20, np.nan, 40])
    assert not validate_bbox([10, 20, np.inf, 40])


# Test for compute_iou_matrix function
def test_compute_iou_matrix_empty():
    evaluator = create_test_evaluator()
    # Test with empty arrays
    result = evaluator.compute_iou_matrix(np.array([]), np.array([]))
    assert len(result) == 0


def test_compute_iou_matrix_no_overlap():
    evaluator = create_test_evaluator()
    # Test with non-overlapping boxes
    dt_boxes = np.array([[0, 0, 10, 10]])
    gt_boxes = np.array([[20, 20, 10, 10]])
    result = evaluator.compute_iou_matrix(dt_boxes, gt_boxes)
    assert result.shape == (1, 1)
    assert result[0, 0] == 0


def test_compute_iou_matrix_full_overlap():
    evaluator = create_test_evaluator()
    # Test with fully overlapping boxes
    dt_boxes = np.array([[10, 10, 10, 10]])
    gt_boxes = np.array([[10, 10, 10, 10]])
    result = evaluator.compute_iou_matrix(dt_boxes, gt_boxes)
    assert result.shape == (1, 1)
    assert result[0, 0] == 1.0


def test_compute_iou_matrix_partial_overlap():
    evaluator = create_test_evaluator()
    # Test with partially overlapping boxes
    dt_boxes = np.array([[0, 0, 20, 20]])
    gt_boxes = np.array([[10, 10, 20, 20]])
    result = evaluator.compute_iou_matrix(dt_boxes, gt_boxes)
    assert result.shape == (1, 1)
    # Overlap area is 10x10 = 100
    # Union area is 20x20 + 20x20 - 100 = 700
    # IoU = 100/700 = 1/7 ≈ 0.143
    assert np.isclose(result[0, 0], 100 / 700)


def test_compute_iou_matrix_multiple_boxes():
    evaluator = create_test_evaluator()
    # Test with multiple boxes
    dt_boxes = np.array([[0, 0, 10, 10], [20, 20, 10, 10]])
    gt_boxes = np.array([[5, 5, 10, 10], [25, 25, 10, 10]])
    result = evaluator.compute_iou_matrix(dt_boxes, gt_boxes)
    assert result.shape == (2, 2)
    # First dt box overlaps with first gt box
    # Overlap area is 5x5 = 25
    # Union area is 10x10 + 10x10 - 25 = 175
    # IoU = 25/175 ≈ 0.143
    assert np.isclose(result[0, 0], 25 / 175)
    # Second dt box overlaps with second gt box
    # Similar calculation gives IoU = 25/175
    assert np.isclose(result[1, 1], 25 / 175)
    # No overlap between first dt and second gt, or second dt and first gt
    assert result[0, 1] == 0
    assert result[1, 0] == 0


# Test for evaluate_category function
def test_evaluate_category_empty():
    evaluator = create_test_evaluator()
    # Test with empty predictions and ground truth
    predictions_df = pl.DataFrame(schema={"image_id": pl.Int64, "category_id": pl.Int64, "bbox": pl.List(pl.Float64), "score": pl.Float64})
    gt_df = pl.DataFrame(schema={"image_id": pl.Int64, "category_id": pl.Int64, "bbox": pl.List(pl.Float64)})
    tp, fp, fn = evaluator.evaluate_category(predictions_df, gt_df, 0.5)
    assert tp == 0
    assert fp == 0
    assert fn == 0


def test_evaluate_category_no_predictions():
    evaluator = create_test_evaluator()
    # Test with no predictions but some ground truth
    predictions_df = pl.DataFrame(schema={"image_id": pl.Int64, "category_id": pl.Int64, "bbox": pl.List(pl.Float64), "score": pl.Float64})
    gt_df = pl.DataFrame({"image_id": [1, 1], "category_id": [0, 0], "bbox": [[0, 0, 10, 10], [20, 20, 10, 10]]})
    tp, fp, fn = evaluator.evaluate_category(predictions_df, gt_df, 0.5)
    assert tp == 0
    assert fp == 0
    assert fn == 2  # Two ground truth boxes, no predictions


def test_evaluate_category_no_ground_truth():
    evaluator = create_test_evaluator()
    # Test with predictions but no ground truth
    predictions_df = pl.DataFrame({"image_id": [1, 1], "category_id": [0, 0], "bbox": [[0, 0, 10, 10], [20, 20, 10, 10]], "score": [0.9, 0.8]})
    gt_df = pl.DataFrame(schema={"image_id": pl.Int64, "category_id": pl.Int64, "bbox": pl.List(pl.Float64)})
    tp, fp, fn = evaluator.evaluate_category(predictions_df, gt_df, 0.5)
    assert tp == 0
    assert fp == 2  # Two predictions, no ground truth
    assert fn == 0


def test_evaluate_category_perfect_match():
    evaluator = create_test_evaluator()
    # Test with perfect match between predictions and ground truth
    predictions_df = pl.DataFrame({"image_id": [1, 1], "category_id": [0, 0], "bbox": [[0, 0, 10, 10], [20, 20, 10, 10]], "score": [0.9, 0.8]})
    gt_df = pl.DataFrame({"image_id": [1, 1], "category_id": [0, 0], "bbox": [[0, 0, 10, 10], [20, 20, 10, 10]]})
    tp, fp, fn = evaluator.evaluate_category(predictions_df, gt_df, 0.5)
    assert tp == 2  # Two perfect matches
    assert fp == 0
    assert fn == 0


def test_evaluate_category_partial_match():
    evaluator = create_test_evaluator()
    # Test with partial matches
    predictions_df = pl.DataFrame(
        {"image_id": [1, 1, 1], "category_id": [0, 0, 0], "bbox": [[2, 2, 10, 10], [20, 20, 10, 10], [40, 40, 10, 10]], "score": [0.9, 0.8, 0.7]}
    )
    gt_df = pl.DataFrame({"image_id": [1, 1], "category_id": [0, 0], "bbox": [[0, 0, 10, 10], [20, 20, 10, 10]]})
    # Use a lower IoU threshold to match the first prediction with the first ground truth box
    tp, fp, fn = evaluator.evaluate_category(predictions_df, gt_df, 0.2)
    # The first prediction overlaps with first ground truth
    # The second prediction perfectly matches second ground truth
    # The third prediction has no match
    assert tp == 2  # Two matches
    assert fp == 1  # One false positive (third prediction)
    assert fn == 0  # All ground truth matched


def test_evaluate_category_multiple_images():
    evaluator = create_test_evaluator()
    # Test with multiple images
    predictions_df = pl.DataFrame(
        {
            "image_id": [1, 1, 2, 2],
            "category_id": [0, 0, 0, 0],
            "bbox": [[0, 0, 10, 10], [20, 20, 10, 10], [0, 0, 10, 10], [20, 20, 10, 10]],
            "score": [0.9, 0.8, 0.7, 0.6],
        }
    )
    gt_df = pl.DataFrame({"image_id": [1, 2, 3], "category_id": [0, 0, 0], "bbox": [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]]})
    tp, fp, fn = evaluator.evaluate_category(predictions_df, gt_df, 0.5)
    # Image 1: 1 TP, 1 FP
    # Image 2: 1 TP, 1 FP
    # Image 3: 1 FN (no predictions for this image)
    assert tp == 2
    assert fp == 2
    assert fn == 1


# Test for the metrics calculation functions
def test_calculate_metrics_at_current_thresholds_empty():
    evaluator = create_test_evaluator_empty()
    # Test with empty predictions
    result = evaluator.calculate_metrics_at_current_thresholds()
    assert result.shape[0] == 0  # No results for empty predictions


def test_calculate_metrics_at_current_thresholds():
    evaluator = create_test_evaluator_with_data()
    # Test with actual predictions and ground truth
    result = evaluator.calculate_metrics_at_current_thresholds(beta=1, iou_thr=0.5)
    # Verify the structure and contents of the result
    assert "class_name" in result.columns
    assert "precision" in result.columns
    assert "recall" in result.columns
    assert "fbeta" in result.columns
    assert "tp" in result.columns
    assert "fp" in result.columns
    assert "fn" in result.columns

    # Check specific metrics for test class
    test_class_metrics = result.filter(pl.col("class_name") == "test_class")
    assert test_class_metrics.shape[0] == 1
    assert 0 <= test_class_metrics["precision"].item() <= 1
    assert 0 <= test_class_metrics["recall"].item() <= 1
    assert 0 <= test_class_metrics["fbeta"].item() <= 1


def test_calculate_fbeta_over_thresholds():
    evaluator = create_test_evaluator_with_data()
    # Test calculation of F-beta over thresholds
    result_df = evaluator._calculate_fbeta_over_thresholds(beta=1, conf_thresholds=np.array([0.1, 0.5, 0.9]))

    # The method returns a DataFrame, not populating eval_f_beta
    assert isinstance(result_df, pl.DataFrame)
    assert len(result_df) > 0

    # Check that the DataFrame has the expected columns
    expected_columns = ["class_id", "conf_thr", "precision", "recall", "fbeta", "tp", "fp", "fn"]
    for col in expected_columns:
        assert col in result_df.columns


def test_calculate_metrics_at_best_thresholds():
    evaluator = create_test_evaluator_with_data()
    # First calculate F-beta over thresholds
    result_df = evaluator._calculate_fbeta_over_thresholds(beta=1, conf_thresholds=np.array([0.1, 0.5, 0.9]))

    # Manually populate eval_f_beta since the method doesn't do it
    evaluator.eval_f_beta[0.5] = result_df.to_dicts()

    # Then test calculation of metrics at best thresholds
    result = evaluator.calculate_metrics_at_best_thresholds(beta=1)
    # Verify the structure and contents of the result
    assert "name" in result.columns
    assert "best_fbeta" in result.columns
    assert "conf" in result.columns
    assert "precision" in result.columns
    assert "recall" in result.columns
    assert "tp" in result.columns
    assert "fp" in result.columns
    assert "fn" in result.columns

    # Check specific metrics for test class
    test_class_metrics = result.filter(pl.col("name") == "test_class")
    assert test_class_metrics.shape[0] == 1
    assert 0 <= test_class_metrics["precision"].item() <= 1
    assert 0 <= test_class_metrics["recall"].item() <= 1
    assert 0 <= test_class_metrics["best_fbeta"].item() <= 1


def test_calculate_metrics_at_best_thresholds_no_ground_truth():
    evaluator = create_test_evaluator_with_no_ground_truth()
    # First calculate F-beta over thresholds
    result_df = evaluator._calculate_fbeta_over_thresholds(beta=1, conf_thresholds=np.array([0.1, 0.5, 0.9]))

    # Manually populate eval_f_beta since the method doesn't do it
    evaluator.eval_f_beta[0.5] = result_df.to_dicts()

    # Then test calculation of metrics at best thresholds
    result = evaluator.calculate_metrics_at_best_thresholds(beta=1)
    # For classes with no ground truth, it should choose threshold that minimizes FP
    # This is typically the highest confidence threshold
    test_class_metrics = result.filter(pl.col("name") == "test_class")
    if not test_class_metrics.is_empty():
        # Should choose highest confidence threshold to minimize FP
        assert test_class_metrics["conf"].item() == 0.9
        assert test_class_metrics["tp"].item() == 0
        assert test_class_metrics["fn"].item() == 0


# Helper functions to create test evaluators
def create_test_evaluator():
    """Create a test evaluator with minimal data."""
    # Create minimal dataframes
    annotations_df = pl.DataFrame(
        {"class_name": ["test_class"], "file_name": ["test_image.jpg"], "box_x1": [0.0], "box_y1": [0.0], "box_width": [10.0], "box_height": [10.0]}
    )

    images_df = pl.DataFrame({"image_id": [1], "file_name": ["test_image.jpg"], "frame_uri": ["gs://test-bucket/test_image.jpg"]})

    predictions_df = pl.DataFrame({"image_id": [1], "class_name": ["test_class"], "bbox": [[0.0, 0.0, 10.0, 10.0]], "score": [0.9]})

    model_class_list = ["test_class"]

    return Evaluator(annotations_df=annotations_df, images_df=images_df, predictions_df=predictions_df, model_class_list=model_class_list)


def create_test_evaluator_empty():
    """Create a test evaluator with empty predictions."""
    # Create minimal dataframes
    annotations_df = pl.DataFrame(
        {"class_name": ["test_class"], "file_name": ["test_image.jpg"], "box_x1": [0.0], "box_y1": [0.0], "box_width": [10.0], "box_height": [10.0]}
    )

    images_df = pl.DataFrame({"image_id": [1], "file_name": ["test_image.jpg"], "frame_uri": ["gs://test-bucket/test_image.jpg"]})

    predictions_df = pl.DataFrame(schema={"image_id": pl.Int64, "class_name": pl.Utf8, "bbox": pl.List(pl.Float64), "score": pl.Float64})

    model_class_list = ["test_class"]

    return Evaluator(annotations_df=annotations_df, images_df=images_df, predictions_df=predictions_df, model_class_list=model_class_list)


def create_test_evaluator_with_data():
    """Create a test evaluator with more complete data for testing metrics."""
    # Create ground truth annotations
    annotations_df = pl.DataFrame(
        {
            "class_name": ["test_class", "test_class", "test_class", "other_class"],
            "file_name": ["image1.jpg", "image1.jpg", "image2.jpg", "image2.jpg"],
            "box_x1": [0.0, 100.0, 0.0, 100.0],
            "box_y1": [0.0, 100.0, 0.0, 100.0],
            "box_width": [50.0, 50.0, 50.0, 50.0],
            "box_height": [50.0, 50.0, 50.0, 50.0],
        }
    )

    # Create image metadata
    images_df = pl.DataFrame(
        {"image_id": [1, 2], "file_name": ["image1.jpg", "image2.jpg"], "frame_uri": ["gs://test-bucket/image1.jpg", "gs://test-bucket/image2.jpg"]}
    )

    # Create predictions with various confidence scores
    predictions_df = pl.DataFrame(
        {
            "image_id": [1, 1, 2, 2],
            "class_name": ["test_class", "test_class", "test_class", "other_class"],
            "bbox": [[10.0, 10.0, 50.0, 50.0], [110.0, 110.0, 50.0, 50.0], [10.0, 10.0, 50.0, 50.0], [110.0, 110.0, 50.0, 50.0]],
            "score": [0.9, 0.7, 0.8, 0.6],
        }
    )

    model_class_list = ["test_class", "other_class"]

    return Evaluator(annotations_df=annotations_df, images_df=images_df, predictions_df=predictions_df, model_class_list=model_class_list)


def create_test_evaluator_with_no_ground_truth():
    """Create a test evaluator with predictions but no ground truth for a class."""
    # Create ground truth annotations (none for test_class)
    annotations_df = pl.DataFrame(
        {
            "class_name": ["other_class", "other_class"],
            "file_name": ["image1.jpg", "image2.jpg"],
            "box_x1": [0.0, 100.0],
            "box_y1": [0.0, 100.0],
            "box_width": [50.0, 50.0],
            "box_height": [50.0, 50.0],
        }
    )

    # Create image metadata
    images_df = pl.DataFrame(
        {"image_id": [1, 2], "file_name": ["image1.jpg", "image2.jpg"], "frame_uri": ["gs://test-bucket/image1.jpg", "gs://test-bucket/image2.jpg"]}
    )

    # Create predictions for test_class (which has no ground truth)
    predictions_df = pl.DataFrame(
        {
            "image_id": [1, 2, 1, 2],
            "class_name": ["test_class", "test_class", "other_class", "other_class"],
            "bbox": [[10.0, 10.0, 50.0, 50.0], [110.0, 110.0, 50.0, 50.0], [10.0, 10.0, 50.0, 50.0], [110.0, 110.0, 50.0, 50.0]],
            "score": [0.5, 0.3, 0.9, 0.7],
        }
    )

    model_class_list = ["test_class", "other_class"]

    return Evaluator(annotations_df=annotations_df, images_df=images_df, predictions_df=predictions_df, model_class_list=model_class_list)
