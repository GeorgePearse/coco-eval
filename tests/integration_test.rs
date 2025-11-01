//! Integration tests for the complete COCO evaluation pipeline.

use coco_eval::evaluator::evaluate;
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

#[test]
fn test_perfect_predictions() {
    // Perfect predictions should give mAP = 1.0
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
            create_annotation(2, 1, 1, vec![100.0, 100.0, 50.0, 50.0], None),
        ],
        categories: vec![create_category(1, "person")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.95)),
            create_annotation(2, 1, 1, vec![100.0, 100.0, 50.0, 50.0], Some(0.90)),
        ],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    assert!(metrics.map > 0.99, "mAP should be ~1.0 for perfect predictions, got {}", metrics.map);
    assert!(metrics.ap50 > 0.99, "AP50 should be ~1.0, got {}", metrics.ap50);
    assert!(metrics.ap75 > 0.99, "AP75 should be ~1.0, got {}", metrics.ap75);
}

#[test]
fn test_no_predictions() {
    // No predictions should give mAP = 0.0
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
        ],
        categories: vec![create_category(1, "person")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    assert_eq!(metrics.map, 0.0, "mAP should be 0.0 with no predictions");
    assert_eq!(metrics.ap50, 0.0, "AP50 should be 0.0");
    assert_eq!(metrics.ap75, 0.0, "AP75 should be 0.0");
}

#[test]
fn test_all_false_positives() {
    // Predictions with no matching ground truth
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
        ],
        categories: vec![create_category(1, "person")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![200.0, 200.0, 50.0, 50.0], Some(0.9)),
        ],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    assert_eq!(metrics.map, 0.0, "mAP should be 0.0 with all false positives");
}

#[test]
fn test_multi_class_evaluation() {
    // Test that each class is evaluated independently
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
            create_annotation(2, 1, 2, vec![100.0, 100.0, 50.0, 50.0], None),
        ],
        categories: vec![
            create_category(1, "person"),
            create_category(2, "car"),
        ],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![
            // Perfect match for person
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.95)),
            // No match for car (wrong location)
            create_annotation(2, 1, 2, vec![200.0, 200.0, 50.0, 50.0], Some(0.90)),
        ],
        categories: vec![
            create_category(1, "person"),
            create_category(2, "car"),
        ],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    // mAP should be 0.5 (1.0 for person, 0.0 for car)
    assert!(metrics.map > 0.4 && metrics.map < 0.6, "mAP should be ~0.5, got {}", metrics.map);

    // Check per-class AP
    assert_eq!(metrics.ap_per_class.len(), 2);

    let person_ap = metrics.ap_per_class.iter().find(|&&(id, _)| id == 1).map(|&(_, ap)| ap);
    let car_ap = metrics.ap_per_class.iter().find(|&&(id, _)| id == 2).map(|&(_, ap)| ap);

    assert!(person_ap.unwrap() > 0.9, "Person AP should be ~1.0");
    assert_eq!(car_ap.unwrap(), 0.0, "Car AP should be 0.0");
}

#[test]
fn test_confidence_thresholding() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
            create_annotation(2, 1, 1, vec![100.0, 100.0, 50.0, 50.0], None),
        ],
        categories: vec![create_category(1, "person")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.95)),
            create_annotation(2, 1, 1, vec![100.0, 100.0, 50.0, 50.0], Some(0.90)),
            create_annotation(3, 1, 1, vec![200.0, 200.0, 50.0, 50.0], Some(0.30)), // FP
        ],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    // Check precision at different thresholds
    assert!(!metrics.precision_at_thresholds.is_empty());
    assert!(!metrics.recall_at_thresholds.is_empty());
    assert!(!metrics.f1_at_thresholds.is_empty());

    // At high confidence (0.9), precision should be high
    let high_conf_precision = metrics
        .precision_at_thresholds
        .iter()
        .find(|(t, _)| (*t - 0.9).abs() < 0.01)
        .map(|(_, p)| *p);
    assert!(high_conf_precision.unwrap() > 0.9, "High confidence precision should be >0.9");

    // At low confidence (0.0), recall should be high but precision lower
    let low_conf_recall = metrics
        .recall_at_thresholds
        .iter()
        .find(|(t, _)| *t < 0.01)
        .map(|(_, r)| *r);
    assert!(low_conf_recall.unwrap() > 0.9, "Low confidence recall should be high");
}

#[test]
fn test_iou_threshold_sensitivity() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
        ],
        categories: vec![create_category(1, "person")],
    };

    // Prediction with moderate overlap
    let predictions = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![15.0, 15.0, 50.0, 50.0], Some(0.95)),
        ],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    // AP50 should be higher than AP75 for moderate overlap
    assert!(metrics.ap50 >= metrics.ap75, "AP50 should be >= AP75 for moderate overlap");
}

#[test]
fn test_multiple_images() {
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
            create_annotation(2, 2, 1, vec![20.0, 20.0, 50.0, 50.0], None),
            create_annotation(3, 3, 1, vec![30.0, 30.0, 50.0, 50.0], None),
        ],
        categories: vec![create_category(1, "person")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.95)),
            create_annotation(2, 2, 1, vec![20.0, 20.0, 50.0, 50.0], Some(0.90)),
            create_annotation(3, 3, 1, vec![30.0, 30.0, 50.0, 50.0], Some(0.85)),
        ],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    assert!(metrics.map > 0.99, "mAP should be ~1.0 for perfect predictions across multiple images");
}

#[test]
fn test_false_positive_before_true_positive() {
    // False positive with higher confidence than true positive should reduce AP
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
            create_annotation(2, 1, 1, vec![100.0, 100.0, 50.0, 50.0], None),
        ],
        categories: vec![create_category(1, "person")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![
            create_annotation(1, 1, 1, vec![200.0, 200.0, 50.0, 50.0], Some(0.95)), // FP (high conf)
            create_annotation(2, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.90)),   // TP
            create_annotation(3, 1, 1, vec![100.0, 100.0, 50.0, 50.0], Some(0.85)), // TP
        ],
        categories: vec![create_category(1, "person")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    // FP before TPs should reduce AP below 1.0
    assert!(metrics.map < 1.0, "mAP should be <1.0 when FP comes before TP, got {}", metrics.map);
    assert!(metrics.map > 0.5, "mAP should still be >0.5, got {}", metrics.map);
}
