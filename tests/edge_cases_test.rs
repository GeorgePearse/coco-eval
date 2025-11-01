//! Comprehensive edge case and boundary condition tests.

use coco_eval::evaluator::evaluate;
use coco_eval::matching::{group_annotations, match_detections};
use coco_eval::types::{Annotation, Category, CocoDataset};

fn create_annotation(
    id: u64,
    image_id: u64,
    category_id: u64,
    bbox: Vec<f64>,
    score: Option<f64>,
) -> Annotation {
    let area = Some(bbox[2] * bbox[3]);
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
// MATCHING EDGE CASES
// ============================================================================

#[test]
fn test_empty_predictions_with_ground_truth() {
    let ground_truth = vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None)];
    let predictions = vec![];

    let matches = match_detections(&predictions, &ground_truth, 0.5).unwrap();
    assert_eq!(matches.len(), 0, "Should have no matches with empty predictions");
}

#[test]
fn test_empty_ground_truth_with_predictions() {
    let ground_truth = vec![];
    let predictions = vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.9))];

    let matches = match_detections(&predictions, &ground_truth, 0.5).unwrap();
    assert_eq!(matches.len(), 1, "Should have one match");
    assert!(!matches[0].is_true_positive, "Should be false positive");
}

#[test]
fn test_many_predictions_one_ground_truth() {
    let ground_truth = vec![create_annotation(1, 1, 1, vec![50.0, 50.0, 100.0, 100.0], None)];

    let predictions = vec![
        create_annotation(1, 1, 1, vec![50.0, 50.0, 100.0, 100.0], Some(0.95)),
        create_annotation(2, 1, 1, vec![52.0, 52.0, 100.0, 100.0], Some(0.90)),
        create_annotation(3, 1, 1, vec![48.0, 48.0, 100.0, 100.0], Some(0.85)),
        create_annotation(4, 1, 1, vec![55.0, 55.0, 100.0, 100.0], Some(0.80)),
    ];

    let matches = match_detections(&predictions, &ground_truth, 0.5).unwrap();
    assert_eq!(matches.len(), 4);

    // Only highest confidence should be TP
    let tp_count = matches.iter().filter(|m| m.is_true_positive).count();
    assert_eq!(tp_count, 1, "Only one prediction should match the GT");

    // First match (highest confidence) should be TP
    assert!(matches[0].is_true_positive);
    assert_eq!(matches[0].confidence, 0.95);
}

#[test]
fn test_one_prediction_many_ground_truths() {
    let ground_truth = vec![
        create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
        create_annotation(2, 1, 1, vec![100.0, 100.0, 50.0, 50.0], None),
        create_annotation(3, 1, 1, vec![200.0, 200.0, 50.0, 50.0], None),
    ];

    let predictions = vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.9))];

    let matches = match_detections(&predictions, &ground_truth, 0.5).unwrap();
    assert_eq!(matches.len(), 1);
    assert!(matches[0].is_true_positive, "Should match one GT");
}

#[test]
fn test_zero_area_bounding_boxes() {
    let ground_truth = vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 0.0, 0.0], None)];
    let predictions = vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 0.0, 0.0], Some(0.9))];

    // Should not panic, should handle gracefully
    let matches = match_detections(&predictions, &ground_truth, 0.5).unwrap();
    assert_eq!(matches.len(), 1);
}

#[test]
fn test_very_small_iou_threshold() {
    let ground_truth = vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None)];
    let predictions = vec![create_annotation(1, 1, 1, vec![11.0, 11.0, 50.0, 50.0], Some(0.9))];

    let matches = match_detections(&predictions, &ground_truth, 0.01).unwrap();
    assert!(matches[0].is_true_positive, "Should match with very low threshold");
}

#[test]
fn test_very_high_iou_threshold() {
    let ground_truth = vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None)];
    let predictions = vec![create_annotation(1, 1, 1, vec![11.0, 11.0, 50.0, 50.0], Some(0.9))];

    let matches = match_detections(&predictions, &ground_truth, 0.99).unwrap();
    assert!(!matches[0].is_true_positive, "Should not match with very high threshold");
}

#[test]
fn test_identical_confidence_scores() {
    let ground_truth = vec![
        create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
        create_annotation(2, 1, 1, vec![100.0, 100.0, 50.0, 50.0], None),
    ];

    let predictions = vec![
        create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.9)),
        create_annotation(2, 1, 1, vec![100.0, 100.0, 50.0, 50.0], Some(0.9)),
    ];

    let matches = match_detections(&predictions, &ground_truth, 0.5).unwrap();
    assert_eq!(matches.len(), 2);

    // Both should be TP
    let tp_count = matches.iter().filter(|m| m.is_true_positive).count();
    assert_eq!(tp_count, 2);
}

// ============================================================================
// EVALUATION EDGE CASES
// ============================================================================

#[test]
fn test_single_class_single_image_single_annotation() {
    let dataset = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.9))],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&dataset, &dataset, None, None).unwrap();
    assert!(metrics.map > 0.99, "Should have perfect mAP");
}

#[test]
fn test_predictions_all_below_threshold() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None)],
        categories: vec![create_category(1, "person")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![200.0, 200.0, 50.0, 50.0], Some(0.01))],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    // At high confidence thresholds, should have zero precision/recall
    let high_conf_precision = metrics
        .precision_at_thresholds
        .iter()
        .find(|(t, _)| *t > 0.5)
        .map(|(_, p)| *p);

    assert_eq!(high_conf_precision.unwrap(), 0.0, "Should have zero precision at high threshold");
}

#[test]
fn test_100_classes() {
    let mut gt_annotations = Vec::new();
    let mut pred_annotations = Vec::new();
    let mut categories = Vec::new();

    for class_id in 1..=100 {
        categories.push(create_category(class_id, &format!("class_{}", class_id)));

        let x = (class_id - 1) as f64 * 10.0;
        gt_annotations.push(create_annotation(
            class_id,
            1,
            class_id,
            vec![x, 10.0, 50.0, 50.0],
            None,
        ));

        pred_annotations.push(create_annotation(
            class_id,
            1,
            class_id,
            vec![x, 10.0, 50.0, 50.0],
            Some(0.9),
        ));
    }

    let ground_truth = CocoDataset {
        images: None,
        annotations: gt_annotations,
        categories: categories.clone(),
    };

    let predictions = CocoDataset {
        images: None,
        annotations: pred_annotations,
        categories,
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    assert_eq!(metrics.ap_per_class.len(), 100, "Should have AP for all 100 classes");
    assert!(metrics.map > 0.99, "Should have perfect mAP for all classes");
}

#[test]
fn test_100_images() {
    let mut gt_annotations = Vec::new();
    let mut pred_annotations = Vec::new();

    for image_id in 1..=100 {
        gt_annotations.push(create_annotation(
            image_id,
            image_id,
            1,
            vec![10.0, 10.0, 50.0, 50.0],
            None,
        ));

        pred_annotations.push(create_annotation(
            image_id,
            image_id,
            1,
            vec![10.0, 10.0, 50.0, 50.0],
            Some(0.9),
        ));
    }

    let ground_truth = CocoDataset {
        images: None,
        annotations: gt_annotations,
        categories: vec![create_category(1, "person")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: pred_annotations,
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();
    assert!(metrics.map > 0.99, "Should have perfect mAP across 100 images");
}

#[test]
fn test_mixed_quality_predictions() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
            create_annotation(2, 1, 1, vec![100.0, 100.0, 50.0, 50.0], None),
            create_annotation(3, 1, 1, vec![200.0, 200.0, 50.0, 50.0], None),
            create_annotation(4, 1, 1, vec![300.0, 300.0, 50.0, 50.0], None),
        ],
        categories: vec![create_category(1, "person")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![
            // Perfect match
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.95)),
            // Good match
            create_annotation(2, 1, 1, vec![102.0, 102.0, 50.0, 50.0], Some(0.90)),
            // Poor match (low IoU)
            create_annotation(3, 1, 1, vec![220.0, 220.0, 50.0, 50.0], Some(0.85)),
            // Complete miss
            create_annotation(4, 1, 1, vec![400.0, 400.0, 50.0, 50.0], Some(0.80)),
        ],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    // AP50 should be >= AP75 (lower IoU thresholds are more lenient)
    assert!(
        metrics.ap50 >= metrics.ap75,
        "AP50 ({}) should be >= AP75 ({})",
        metrics.ap50,
        metrics.ap75
    );
}

#[test]
fn test_category_mismatch() {
    // Predictions for wrong category should not match
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None)],
        categories: vec![create_category(1, "person"), create_category(2, "car")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![
            // Same location but wrong category
            create_annotation(1, 1, 2, vec![10.0, 10.0, 50.0, 50.0], Some(0.95)),
        ],
        categories: vec![create_category(1, "person"), create_category(2, "car")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    // Person class should have AP=0 (no predictions)
    let person_ap = metrics
        .ap_per_class
        .iter()
        .find(|&&(id, _)| id == 1)
        .map(|&(_, ap)| ap);
    assert_eq!(person_ap.unwrap(), 0.0, "Person class should have AP=0");

    // Car class should have AP=0 (FP only)
    let car_ap = metrics
        .ap_per_class
        .iter()
        .find(|&&(id, _)| id == 2)
        .map(|&(_, ap)| ap);
    assert_eq!(car_ap.unwrap(), 0.0, "Car class should have AP=0");
}

#[test]
fn test_varying_iou_thresholds() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None)],
        categories: vec![create_category(1, "person")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![
            // Moderate overlap
            create_annotation(1, 1, 1, vec![20.0, 20.0, 50.0, 50.0], Some(0.9)),
        ],
        categories: vec![create_category(1, "person")],
    };

    // Test with single IoU threshold
    let metrics_50 = evaluate(&ground_truth, &predictions, Some(vec![0.5]), None).unwrap();
    let metrics_75 = evaluate(&ground_truth, &predictions, Some(vec![0.75]), None).unwrap();

    // AP at IoU=0.5 should be higher than at IoU=0.75
    assert!(
        metrics_50.map >= metrics_75.map,
        "AP at lower IoU should be >= AP at higher IoU"
    );
}

#[test]
fn test_group_annotations_edge_cases() {
    // Empty annotations
    let groups = group_annotations(&[]);
    assert_eq!(groups.len(), 0);

    // Single annotation
    let single = vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None)];
    let groups = group_annotations(&single);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[&(1, 1)].len(), 1);

    // Multiple images, single category
    let multi_image = vec![
        create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
        create_annotation(2, 2, 1, vec![10.0, 10.0, 50.0, 50.0], None),
        create_annotation(3, 3, 1, vec![10.0, 10.0, 50.0, 50.0], None),
    ];
    let groups = group_annotations(&multi_image);
    assert_eq!(groups.len(), 3);

    // Single image, multiple categories
    let multi_cat = vec![
        create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
        create_annotation(2, 1, 2, vec![10.0, 10.0, 50.0, 50.0], None),
        create_annotation(3, 1, 3, vec![10.0, 10.0, 50.0, 50.0], None),
    ];
    let groups = group_annotations(&multi_cat);
    assert_eq!(groups.len(), 3);
}

#[test]
fn test_extreme_confidence_values() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None)],
        categories: vec![create_category(1, "person")],
    };

    // Test with confidence = 1.0
    let pred_high = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(1.0))],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &pred_high, None, None).unwrap();
    assert!(metrics.map > 0.99);

    // Test with confidence = 0.0
    let pred_low = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.0))],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &pred_low, None, None).unwrap();
    // At threshold 0.0, should still match
    assert!(metrics.map > 0.99);
}
