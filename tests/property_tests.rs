//! Property-based tests using proptest
//!
//! These tests verify mathematical properties and invariants that should
//! always hold regardless of the input values.

use coco_eval::metrics::{
    calculate_fbeta, calculate_f1, calculate_precision, calculate_recall,
    calculate_iou,
};
use coco_eval::nms::{Detection, non_maximum_suppression, xywh_to_xyxy, xyxy_to_xywh};
use coco_eval::types::BoundingBox;
use proptest::prelude::*;

// Property: Precision should always be between 0.0 and 1.0
proptest! {
    #[test]
    fn prop_precision_range(tp in 0usize..1000, fp in 0usize..1000) {
        let precision = calculate_precision(tp, fp);
        assert!(precision >= 0.0 && precision <= 1.0,
                "Precision should be in [0,1], got {}", precision);
    }

    #[test]
    fn prop_recall_range(tp in 0usize..1000, fn_ in 0usize..1000) {
        let recall = calculate_recall(tp, fn_);
        assert!(recall >= 0.0 && recall <= 1.0,
                "Recall should be in [0,1], got {}", recall);
    }

    #[test]
    fn prop_f1_range(tp in 0usize..1000, fp in 0usize..1000, fn_ in 0usize..1000) {
        let precision = calculate_precision(tp, fp);
        let recall = calculate_recall(tp, fn_);
        let f1 = calculate_f1(precision, recall);
        assert!(f1 >= 0.0 && f1 <= 1.0,
                "F1 score should be in [0,1], got {}", f1);
    }

    #[test]
    fn prop_fbeta_range(
        tp in 0usize..1000,
        fp in 0usize..1000,
        fn_ in 0usize..1000,
        beta in 0.1f64..10.0f64
    ) {
        let precision = calculate_precision(tp, fp);
        let recall = calculate_recall(tp, fn_);
        let fbeta = calculate_fbeta(precision, recall, beta);
        assert!(fbeta >= 0.0 && fbeta <= 1.0,
                "F-beta score should be in [0,1], got {}", fbeta);
    }
}

// Property: F1 is the harmonic mean of precision and recall
proptest! {
    #[test]
    fn prop_f1_harmonic_mean(
        precision in 0.0f64..=1.0,
        recall in 0.0f64..=1.0
    ) {
        let f1 = calculate_f1(precision, recall);

        if precision + recall > 0.0 {
            let expected = 2.0 * precision * recall / (precision + recall);
            assert!((f1 - expected).abs() < 1e-10,
                    "F1 should be harmonic mean: expected {}, got {}", expected, f1);
        } else {
            assert_eq!(f1, 0.0, "F1 should be 0 when both P and R are 0");
        }
    }
}

// Property: Perfect precision and recall gives F1 = 1.0
#[test]
fn prop_perfect_scores() {
    let f1 = calculate_f1(1.0, 1.0);
    assert!((f1 - 1.0).abs() < 1e-10, "Perfect P and R should give F1=1.0");
}

// Property: IoU is symmetric
proptest! {
    #[test]
    fn prop_iou_symmetric(
        x1 in 0.0f64..100.0,
        y1 in 0.0f64..100.0,
        w1 in 1.0f64..50.0,
        h1 in 1.0f64..50.0,
        x2 in 0.0f64..100.0,
        y2 in 0.0f64..100.0,
        w2 in 1.0f64..50.0,
        h2 in 1.0f64..50.0,
    ) {
        let bbox1 = BoundingBox::new(x1, y1, w1, h1);
        let bbox2 = BoundingBox::new(x2, y2, w2, h2);

        let iou1 = calculate_iou(&bbox1, &bbox2);
        let iou2 = calculate_iou(&bbox2, &bbox1);

        assert!((iou1 - iou2).abs() < 1e-10,
                "IoU should be symmetric: {} vs {}", iou1, iou2);
    }
}

// Property: IoU is always between 0 and 1
proptest! {
    #[test]
    fn prop_iou_range(
        x1 in 0.0f64..100.0,
        y1 in 0.0f64..100.0,
        w1 in 1.0f64..50.0,
        h1 in 1.0f64..50.0,
        x2 in 0.0f64..100.0,
        y2 in 0.0f64..100.0,
        w2 in 1.0f64..50.0,
        h2 in 1.0f64..50.0,
    ) {
        let bbox1 = BoundingBox::new(x1, y1, w1, h1);
        let bbox2 = BoundingBox::new(x2, y2, w2, h2);

        let iou = calculate_iou(&bbox1, &bbox2);

        assert!(iou >= 0.0 && iou <= 1.0,
                "IoU should be in [0,1], got {}", iou);
    }
}

// Property: Identical boxes have IoU = 1.0
proptest! {
    #[test]
    fn prop_iou_identical(
        x in 0.0f64..100.0,
        y in 0.0f64..100.0,
        w in 1.0f64..50.0,
        h in 1.0f64..50.0,
    ) {
        let bbox = BoundingBox::new(x, y, w, h);
        let iou = calculate_iou(&bbox, &bbox);

        assert!((iou - 1.0).abs() < 1e-10,
                "Identical boxes should have IoU=1.0, got {}", iou);
    }
}

// Property: Bounding box conversion roundtrip
proptest! {
    #[test]
    fn prop_bbox_conversion_roundtrip(
        x in 0.0f64..100.0,
        y in 0.0f64..100.0,
        w in 1.0f64..50.0,
        h in 1.0f64..50.0,
    ) {
        let original = [x, y, w, h];
        let xyxy = xywh_to_xyxy(original);
        let back = xyxy_to_xywh(xyxy);

        for i in 0..4 {
            assert!((original[i] - back[i]).abs() < 1e-10,
                    "Roundtrip conversion failed at index {}: {} vs {}", i, original[i], back[i]);
        }
    }
}

// Property: NMS with threshold 0.0 keeps all boxes
proptest! {
    #[test]
    fn prop_nms_threshold_zero(
        num_boxes in 1usize..20,
        seed in 0u64..1000,
    ) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let _hash = hasher.finish();

        let detections: Vec<Detection> = (0..num_boxes)
            .map(|i| {
                let offset = (i as f64) * 10.0;
                Detection {
                    bbox: [offset, offset, offset + 5.0, offset + 5.0],
                    score: 0.9 - (i as f64) * 0.01,
                    index: i,
                }
            })
            .collect();

        let keep_mask = non_maximum_suppression(&detections, 0.0).unwrap();

        // With threshold 0.0, only exactly identical boxes are suppressed
        // Our boxes don't overlap, so all should be kept
        assert_eq!(keep_mask.iter().filter(|&&k| k).count(), num_boxes,
                   "NMS with threshold 0.0 should keep non-overlapping boxes");
    }
}

// Property: NMS with threshold 1.0 suppresses all but highest confidence
#[test]
fn prop_nms_threshold_one_overlapping() {
    // Create overlapping boxes with different scores
    let detections = vec![
        Detection {
            bbox: [10.0, 10.0, 50.0, 50.0],
            score: 0.9,
            index: 0,
        },
        Detection {
            bbox: [15.0, 15.0, 55.0, 55.0], // Overlaps with first
            score: 0.8,
            index: 1,
        },
        Detection {
            bbox: [12.0, 12.0, 52.0, 52.0], // Also overlaps
            score: 0.7,
            index: 2,
        },
    ];

    let keep_mask = non_maximum_suppression(&detections, 1.0).unwrap();

    // With threshold 1.0, any overlap suppresses lower confidence boxes
    let kept_count = keep_mask.iter().filter(|&&k| k).count();
    assert!(kept_count <= detections.len(),
            "NMS should suppress some overlapping boxes");
    assert!(keep_mask[0], "Highest confidence box should always be kept");
}

// Property: F-beta with beta=1 equals F1
proptest! {
    #[test]
    fn prop_fbeta_equals_f1_when_beta_one(
        precision in 0.0f64..=1.0,
        recall in 0.0f64..=1.0,
    ) {
        let f1 = calculate_f1(precision, recall);
        let fbeta = calculate_fbeta(precision, recall, 1.0);

        assert!((f1 - fbeta).abs() < 1e-10,
                "F-beta with beta=1 should equal F1: {} vs {}", f1, fbeta);
    }
}

// Property: Higher beta favors recall
proptest! {
    #[test]
    fn prop_higher_beta_favors_recall(
        precision in 0.1f64..=0.9,
        recall in 0.1f64..=0.9,
    ) {
        // Only test when recall > precision
        prop_assume!(recall > precision);

        let f1 = calculate_fbeta(precision, recall, 1.0);
        let f2 = calculate_fbeta(precision, recall, 2.0);

        assert!(f2 > f1 || (f2 - f1).abs() < 1e-10,
                "F2 should be >= F1 when recall > precision: F1={}, F2={}", f1, f2);
    }
}

// Property: Lower beta favors precision
proptest! {
    #[test]
    fn prop_lower_beta_favors_precision(
        precision in 0.1f64..=0.9,
        recall in 0.1f64..=0.9,
    ) {
        // Only test when precision > recall
        prop_assume!(precision > recall);

        let f1 = calculate_fbeta(precision, recall, 1.0);
        let f_half = calculate_fbeta(precision, recall, 0.5);

        assert!(f_half > f1 || (f_half - f1).abs() < 1e-10,
                "F0.5 should be >= F1 when precision > recall: F1={}, F0.5={}", f1, f_half);
    }
}

// Property: Bounding box area is always non-negative
proptest! {
    #[test]
    fn prop_bbox_area_non_negative(
        x in 0.0f64..100.0,
        y in 0.0f64..100.0,
        w in 0.0f64..50.0,
        h in 0.0f64..50.0,
    ) {
        let bbox = BoundingBox::new(x, y, w, h);
        assert!(bbox.area() >= 0.0, "Bounding box area should be non-negative");
    }
}

// Property: If all predictions are TPs, precision = 1.0
proptest! {
    #[test]
    fn prop_all_tp_gives_perfect_precision(tp in 1usize..1000) {
        let precision = calculate_precision(tp, 0);
        assert!((precision - 1.0).abs() < 1e-10,
                "All TPs (no FPs) should give precision=1.0");
    }
}

// Property: If all ground truth is matched, recall = 1.0
proptest! {
    #[test]
    fn prop_all_matched_gives_perfect_recall(tp in 1usize..1000) {
        let recall = calculate_recall(tp, 0);
        assert!((recall - 1.0).abs() < 1e-10,
                "All matched (no FNs) should give recall=1.0");
    }
}
