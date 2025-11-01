//! Error handling and validation tests.

use coco_eval::error::CocoEvalError;
use coco_eval::evaluator::evaluate;
use coco_eval::loader::{load_from_string, load_from_file};
use coco_eval::threshold::{filter_by_confidence, generate_threshold_range};
use coco_eval::types::{Annotation, Category, CocoDataset};

fn create_annotation(
    id: u64,
    image_id: u64,
    category_id: u64,
    bbox: Vec<f64>,
    score: Option<f64>,
) -> Annotation {
    let area = if bbox.len() >= 4 {
        Some(bbox[2] * bbox[3])
    } else {
        None
    };
    Annotation {
        id,
        image_id,
        category_id,
        bbox,
        area,
        iscrowd: None,
        score,
    }
}

fn create_category(id: u64, name: &str) -> Category {
    Category {
        id,
        name: name.to_string(),
        supercategory: None,
    }
}

// ============================================================================
// LOADER ERROR TESTS
// ============================================================================

#[test]
fn test_invalid_json() {
    let result = load_from_string("{ invalid json");
    assert!(result.is_err(), "Should fail on invalid JSON");
}

#[test]
fn test_empty_categories_error() {
    let json = r#"{
        "annotations": [],
        "categories": []
    }"#;

    let result = load_from_string(json);
    assert!(result.is_err(), "Should fail with empty categories");

    if let Err(CocoEvalError::EmptyDataset(msg)) = result {
        assert!(msg.contains("category"));
    } else {
        panic!("Expected EmptyDataset error");
    }
}

#[test]
fn test_invalid_bbox_length() {
    let json = r#"{
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 30.0]
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person"
            }
        ]
    }"#;

    let result = load_from_string(json);
    assert!(result.is_err(), "Should fail with invalid bbox length");

    if let Err(CocoEvalError::InvalidAnnotation(msg)) = result {
        assert!(msg.contains("bbox length"));
    } else {
        panic!("Expected InvalidAnnotation error");
    }
}

#[test]
fn test_negative_bbox_dimensions() {
    let json = r#"{
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, -30.0, 40.0]
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person"
            }
        ]
    }"#;

    let result = load_from_string(json);
    assert!(result.is_err(), "Should fail with negative dimensions");

    if let Err(CocoEvalError::InvalidBoundingBox(msg)) = result {
        assert!(msg.contains("negative"));
    } else {
        panic!("Expected InvalidBoundingBox error");
    }
}

#[test]
fn test_missing_file() {
    let result = load_from_file("/path/that/does/not/exist.json");
    assert!(result.is_err(), "Should fail with missing file");
}

// ============================================================================
// THRESHOLD ERROR TESTS
// ============================================================================

#[test]
fn test_threshold_too_high() {
    let annotations = vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.9))];

    let result = filter_by_confidence(&annotations, 1.5);
    assert!(result.is_err(), "Should fail with threshold > 1.0");

    if let Err(CocoEvalError::InvalidThreshold(msg)) = result {
        assert!(msg.contains("between 0.0 and 1.0"));
    } else {
        panic!("Expected InvalidThreshold error");
    }
}

#[test]
fn test_threshold_too_low() {
    let annotations = vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.9))];

    let result = filter_by_confidence(&annotations, -0.1);
    assert!(result.is_err(), "Should fail with threshold < 0.0");
}

#[test]
fn test_threshold_range_zero_steps() {
    let result = generate_threshold_range(0.0, 1.0, 0);
    assert!(result.is_err(), "Should fail with zero steps");

    if let Err(CocoEvalError::InvalidThreshold(msg)) = result {
        assert!(msg.contains("steps"));
    } else {
        panic!("Expected InvalidThreshold error");
    }
}

#[test]
fn test_threshold_range_inverted() {
    let result = generate_threshold_range(1.0, 0.0, 10);
    assert!(result.is_err(), "Should fail when start > end");

    if let Err(CocoEvalError::InvalidThreshold(msg)) = result {
        assert!(msg.contains("<="));
    } else {
        panic!("Expected InvalidThreshold error");
    }
}

#[test]
fn test_threshold_range_single_step() {
    let result = generate_threshold_range(0.5, 0.5, 1);
    assert!(result.is_ok(), "Should succeed with single step");
    assert_eq!(result.unwrap(), vec![0.5]);
}

// ============================================================================
// EVALUATION ERROR TESTS
// ============================================================================

#[test]
fn test_evaluation_with_no_categories() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![],
        categories: vec![],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![],
        categories: vec![],
    };

    let result = evaluate(&ground_truth, &predictions, None, None);
    assert!(result.is_err(), "Should fail with no categories");
}

#[test]
fn test_bbox_conversion_invalid() {
    let ann = create_annotation(1, 1, 1, vec![10.0, 20.0], Some(0.9));
    let result = ann.to_bbox();
    assert!(result.is_err(), "Should fail converting invalid bbox");
}

// ============================================================================
// BOUNDARY VALUE TESTS
// ============================================================================

#[test]
fn test_very_large_bounding_boxes() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![create_annotation(
            1,
            1,
            1,
            vec![0.0, 0.0, 10000.0, 10000.0],
            None,
        )],
        categories: vec![create_category(1, "large")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![create_annotation(
            1,
            1,
            1,
            vec![0.0, 0.0, 10000.0, 10000.0],
            Some(0.9),
        )],
        categories: vec![create_category(1, "large")],
    };

    let result = evaluate(&ground_truth, &predictions, None, None);
    assert!(result.is_ok(), "Should handle very large bounding boxes");
}

#[test]
fn test_very_small_bounding_boxes() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![0.0, 0.0, 0.1, 0.1], None)],
        categories: vec![create_category(1, "tiny")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![0.0, 0.0, 0.1, 0.1], Some(0.9))],
        categories: vec![create_category(1, "tiny")],
    };

    let result = evaluate(&ground_truth, &predictions, None, None);
    assert!(result.is_ok(), "Should handle very small bounding boxes");
}

#[test]
fn test_negative_coordinates() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![create_annotation(
            1,
            1,
            1,
            vec![-100.0, -100.0, 50.0, 50.0],
            None,
        )],
        categories: vec![create_category(1, "negative")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![create_annotation(
            1,
            1,
            1,
            vec![-100.0, -100.0, 50.0, 50.0],
            Some(0.9),
        )],
        categories: vec![create_category(1, "negative")],
    };

    let result = evaluate(&ground_truth, &predictions, None, None);
    assert!(result.is_ok(), "Should handle negative coordinates");
}

#[test]
fn test_floating_point_precision() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![create_annotation(
            1,
            1,
            1,
            vec![
                std::f64::consts::PI,
                std::f64::consts::E,
                std::f64::consts::SQRT_2,
                std::f64::consts::LN_2,
            ],
            None,
        )],
        categories: vec![create_category(1, "math")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![create_annotation(
            1,
            1,
            1,
            vec![
                std::f64::consts::PI,
                std::f64::consts::E,
                std::f64::consts::SQRT_2,
                std::f64::consts::LN_2,
            ],
            Some(0.9),
        )],
        categories: vec![create_category(1, "math")],
    };

    let result = evaluate(&ground_truth, &predictions, None, None);
    assert!(result.is_ok(), "Should handle floating point precision");
}

// ============================================================================
// SPECIAL CHARACTER TESTS
// ============================================================================

#[test]
fn test_special_characters_in_category_names() {
    let dataset = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.9))],
        categories: vec![create_category(1, "category-with-dashes_and_underscores & symbols!")],
    };

    let result = evaluate(&dataset, &dataset, None, None);
    assert!(result.is_ok(), "Should handle special characters in category names");
}

#[test]
fn test_unicode_category_names() {
    let dataset = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.9))],
        categories: vec![create_category(1, "人 🚗 مركبة")],
    };

    let result = evaluate(&dataset, &dataset, None, None);
    assert!(result.is_ok(), "Should handle Unicode in category names");
}

// ============================================================================
// ROBUSTNESS TESTS
// ============================================================================

#[test]
fn test_missing_optional_fields() {
    let json = r#"{
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 30.0, 40.0]
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person"
            }
        ]
    }"#;

    let result = load_from_string(json);
    assert!(result.is_ok(), "Should handle missing optional fields");

    let dataset = result.unwrap();
    assert_eq!(dataset.annotations[0].area, None);
    assert_eq!(dataset.annotations[0].score, None);
    assert_eq!(dataset.annotations[0].iscrowd, None);
}

#[test]
fn test_extra_fields_ignored() {
    let json = r#"{
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 30.0, 40.0],
                "extra_field": "ignored",
                "another_field": 123
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "extra": "also ignored"
            }
        ],
        "extra_top_level": "ignored too"
    }"#;

    let result = load_from_string(json);
    assert!(result.is_ok(), "Should ignore extra fields");
}
