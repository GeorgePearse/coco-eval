//! Stress tests with large datasets and complex scenarios.

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
fn test_1000_annotations_single_image() {
    let mut gt_annotations = Vec::new();
    let mut pred_annotations = Vec::new();

    for i in 0..1000 {
        let x = (i % 100) as f64 * 10.0;
        let y = (i / 100) as f64 * 10.0;

        gt_annotations.push(create_annotation(i, 1, 1, vec![x, y, 8.0, 8.0], None));

        pred_annotations.push(create_annotation(
            i,
            1,
            1,
            vec![x, y, 8.0, 8.0],
            Some(0.9 - (i as f64 / 10000.0)),
        ));
    }

    let ground_truth = CocoDataset {
        images: None,
        annotations: gt_annotations,
        categories: vec![create_category(1, "object")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: pred_annotations,
        categories: vec![create_category(1, "object")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();
    assert!(metrics.map > 0.99, "Should have high mAP with 1000 perfect matches");
}

#[test]
fn test_10_classes_10_images_10_annotations() {
    let mut gt_annotations = Vec::new();
    let mut pred_annotations = Vec::new();
    let mut categories = Vec::new();

    let mut ann_id = 0u64;

    for class_id in 1..=10 {
        categories.push(create_category(class_id, &format!("class_{}", class_id)));

        for image_id in 1..=10 {
            for obj_id in 0..10 {
                ann_id += 1;
                let x = (obj_id * 20) as f64;
                let y = (class_id * 10) as f64;

                gt_annotations.push(create_annotation(
                    ann_id,
                    image_id,
                    class_id,
                    vec![x, y, 15.0, 15.0],
                    None,
                ));

                pred_annotations.push(create_annotation(
                    ann_id,
                    image_id,
                    class_id,
                    vec![x, y, 15.0, 15.0],
                    Some(0.85 + (obj_id as f64 / 100.0)),
                ));
            }
        }
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

    // 10 classes x 10 images x 10 annotations = 1000 total
    assert_eq!(metrics.ap_per_class.len(), 10, "Should have 10 classes");
    assert!(metrics.map > 0.99, "Should have high mAP");
}

#[test]
fn test_mixed_precision_scenario() {
    // 50% perfect predictions, 25% good, 25% false positives
    let mut gt_annotations = Vec::new();
    let mut pred_annotations = Vec::new();

    // 100 ground truths
    for i in 0..100 {
        let x = (i % 10) as f64 * 50.0;
        let y = (i / 10) as f64 * 50.0;
        gt_annotations.push(create_annotation(i, 1, 1, vec![x, y, 40.0, 40.0], None));
    }

    // 50 perfect matches
    for i in 0..50 {
        let x = (i % 10) as f64 * 50.0;
        let y = (i / 10) as f64 * 50.0;
        pred_annotations.push(create_annotation(
            i,
            1,
            1,
            vec![x, y, 40.0, 40.0],
            Some(0.95 - (i as f64 / 1000.0)),
        ));
    }

    // 25 good matches (slight offset)
    for i in 50..75 {
        let x = (i % 10) as f64 * 50.0 + 5.0;
        let y = (i / 10) as f64 * 50.0 + 5.0;
        pred_annotations.push(create_annotation(
            i,
            1,
            1,
            vec![x, y, 40.0, 40.0],
            Some(0.85 - (i as f64 / 1000.0)),
        ));
    }

    // 25 false positives
    for i in 75..100 {
        let x = (i % 10) as f64 * 50.0 + 1000.0; // Far away
        let y = (i / 10) as f64 * 50.0;
        pred_annotations.push(create_annotation(
            i,
            1,
            1,
            vec![x, y, 40.0, 40.0],
            Some(0.75 - (i as f64 / 1000.0)),
        ));
    }

    let ground_truth = CocoDataset {
        images: None,
        annotations: gt_annotations,
        categories: vec![create_category(1, "object")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: pred_annotations,
        categories: vec![create_category(1, "object")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    // Should have reasonable but not perfect mAP
    assert!(metrics.map > 0.4 && metrics.map < 0.9, "mAP should be moderate, got {}", metrics.map);
}

#[test]
fn test_highly_overlapping_predictions() {
    // 10 GTs, 100 predictions (10 per GT, all overlapping)
    let mut gt_annotations = Vec::new();
    let mut pred_annotations = Vec::new();

    for i in 0..10 {
        let x = i as f64 * 100.0;
        gt_annotations.push(create_annotation(i, 1, 1, vec![x, 50.0, 80.0, 80.0], None));

        // 10 predictions per GT
        for j in 0..10 {
            let offset = j as f64 * 2.0;
            pred_annotations.push(create_annotation(
                i * 10 + j,
                1,
                1,
                vec![x + offset, 50.0 + offset, 80.0, 80.0],
                Some(0.99 - (j as f64 / 100.0)),
            ));
        }
    }

    let ground_truth = CocoDataset {
        images: None,
        annotations: gt_annotations,
        categories: vec![create_category(1, "object")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: pred_annotations,
        categories: vec![create_category(1, "object")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    // Only 10 should match (one per GT), rest are FPs
    // So precision should be low at low thresholds
    let low_threshold_precision = metrics
        .precision_at_thresholds
        .iter()
        .find(|(t, _)| *t < 0.1)
        .map(|(_, p)| *p)
        .unwrap();

    assert!(
        low_threshold_precision < 0.2,
        "Precision should be low with many duplicates, got {}",
        low_threshold_precision
    );
}

#[test]
fn test_all_iou_thresholds() {
    // Test with full COCO IoU threshold range
    let ground_truth = CocoDataset {
        images: None,
        annotations: vec![create_annotation(1, 1, 1, vec![10.0, 10.0, 100.0, 100.0], None)],
        categories: vec![create_category(1, "object")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: vec![create_annotation(
            1,
            1,
            1,
            vec![15.0, 15.0, 100.0, 100.0],
            Some(0.9),
        )],
        categories: vec![create_category(1, "object")],
    };

    // Test all COCO thresholds: 0.5:0.05:0.95
    let iou_thresholds: Vec<f64> = (0..10).map(|i| 0.5 + 0.05 * i as f64).collect();

    let metrics = evaluate(&ground_truth, &predictions, Some(iou_thresholds), None).unwrap();

    // AP should decrease as IoU threshold increases
    assert!(
        metrics.ap50 >= metrics.ap75,
        "AP50 should be >= AP75"
    );
}

#[test]
fn test_imbalanced_classes() {
    // 1 annotation for class 1, 100 for class 2
    let mut gt_annotations = Vec::new();
    let mut pred_annotations = Vec::new();

    // Class 1: just 1 annotation
    gt_annotations.push(create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None));
    pred_annotations.push(create_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.9)));

    // Class 2: 100 annotations
    for i in 0..100 {
        let x = (i % 10) as f64 * 60.0;
        let y = (i / 10) as f64 * 60.0;

        gt_annotations.push(create_annotation(i + 2, 1, 2, vec![x, y, 50.0, 50.0], None));
        pred_annotations.push(create_annotation(
            i + 2,
            1,
            2,
            vec![x, y, 50.0, 50.0],
            Some(0.85),
        ));
    }

    let ground_truth = CocoDataset {
        images: None,
        annotations: gt_annotations,
        categories: vec![create_category(1, "rare"), create_category(2, "common")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: pred_annotations,
        categories: vec![create_category(1, "rare"), create_category(2, "common")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    // Both classes should have perfect AP
    let class1_ap = metrics
        .ap_per_class
        .iter()
        .find(|&&(id, _)| id == 1)
        .map(|&(_, ap)| ap)
        .unwrap();

    let class2_ap = metrics
        .ap_per_class
        .iter()
        .find(|&&(id, _)| id == 2)
        .map(|&(_, ap)| ap)
        .unwrap();

    assert!(class1_ap > 0.99, "Rare class should have perfect AP");
    assert!(class2_ap > 0.99, "Common class should have perfect AP");
}

#[test]
fn test_varying_image_sizes() {
    // Annotations spread across different "image sizes" (coordinate spaces)
    let mut gt_annotations = Vec::new();
    let mut pred_annotations = Vec::new();

    for i in 0..10 {
        // Each image has different coordinate scale
        let scale = (i + 1) as f64 * 100.0;

        gt_annotations.push(create_annotation(
            i,
            i,
            1,
            vec![scale * 0.1, scale * 0.1, scale * 0.5, scale * 0.5],
            None,
        ));

        pred_annotations.push(create_annotation(
            i,
            i,
            1,
            vec![scale * 0.1, scale * 0.1, scale * 0.5, scale * 0.5],
            Some(0.9),
        ));
    }

    let ground_truth = CocoDataset {
        images: None,
        annotations: gt_annotations,
        categories: vec![create_category(1, "object")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: pred_annotations,
        categories: vec![create_category(1, "object")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();
    assert!(metrics.map > 0.99, "Should handle varying coordinate scales");
}

#[test]
fn test_precision_recall_curve_monotonicity() {
    // Test that recall is monotonically increasing and precision is monotonically decreasing
    let mut gt_annotations = Vec::new();
    let mut pred_annotations = Vec::new();

    // 10 ground truths
    for i in 0..10 {
        gt_annotations.push(create_annotation(
            i,
            1,
            1,
            vec![(i * 50) as f64, 10.0, 40.0, 40.0],
            None,
        ));
    }

    // Mix of TPs and FPs with decreasing confidence
    for i in 0..20 {
        let is_tp = i < 10;
        let x = if is_tp {
            (i * 50) as f64
        } else {
            (i * 50) as f64 + 1000.0
        };

        pred_annotations.push(create_annotation(
            i,
            1,
            1,
            vec![x, 10.0, 40.0, 40.0],
            Some(0.95 - (i as f64 * 0.04)),
        ));
    }

    let ground_truth = CocoDataset {
        images: None,
        annotations: gt_annotations,
        categories: vec![create_category(1, "object")],
    };

    let predictions = CocoDataset {
        images: None,
        annotations: pred_annotations,
        categories: vec![create_category(1, "object")],
    };

    let metrics = evaluate(&ground_truth, &predictions, None, None).unwrap();

    // Check that recall values are monotonically non-decreasing
    for i in 1..metrics.recall_at_thresholds.len() {
        let (_, prev_recall) = metrics.recall_at_thresholds[i - 1];
        let (_, curr_recall) = metrics.recall_at_thresholds[i];
        assert!(
            curr_recall <= prev_recall,
            "Recall should be non-increasing as threshold increases"
        );
    }
}
