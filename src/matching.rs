//! Detection matching utilities for evaluating predictions against ground truth.

use crate::error::Result;
use crate::metrics::iou::calculate_iou;
use crate::types::{Annotation, BoundingBox};
use std::collections::{HashMap, HashSet};

/// Represents a matched detection with its ground truth.
#[derive(Debug, Clone)]
pub struct Match {
    pub prediction_id: u64,
    pub ground_truth_id: Option<u64>,
    pub iou: f64,
    pub is_true_positive: bool,
    pub confidence: f64,
}

/// Match predictions to ground truth annotations for a single image and category.
///
/// Uses greedy matching: predictions are sorted by confidence (descending),
/// and each prediction is matched to the highest-IoU unmatched ground truth.
///
/// # Arguments
///
/// * `predictions` - Predictions for this image/category
/// * `ground_truths` - Ground truth annotations for this image/category
/// * `iou_threshold` - Minimum IoU to consider a match
///
/// # Returns
///
/// Returns a vector of Match objects, one per prediction, sorted by confidence (descending).
pub fn match_detections(
    predictions: &[Annotation],
    ground_truths: &[Annotation],
    iou_threshold: f64,
) -> Result<Vec<Match>> {
    // Convert annotations to bounding boxes
    let pred_boxes: Vec<BoundingBox> = predictions
        .iter()
        .map(|ann| ann.to_bbox())
        .collect::<Result<Vec<_>>>()?;

    let gt_boxes: Vec<BoundingBox> = ground_truths
        .iter()
        .map(|ann| ann.to_bbox())
        .collect::<Result<Vec<_>>>()?;

    // Create prediction indices sorted by confidence (descending)
    let mut pred_indices: Vec<usize> = (0..predictions.len()).collect();
    pred_indices.sort_by(|&a, &b| {
        predictions[b]
            .confidence()
            .partial_cmp(&predictions[a].confidence())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Track which ground truths have been matched
    let mut matched_gt: HashSet<usize> = HashSet::new();

    // Store matches in original prediction order, then re-sort
    let mut matches_map: HashMap<usize, Match> = HashMap::new();

    // Match each prediction to best available ground truth
    for &pred_idx in &pred_indices {
        let pred = &predictions[pred_idx];
        let pred_box = &pred_boxes[pred_idx];

        let mut best_iou = 0.0;
        let mut best_gt_idx: Option<usize> = None;

        // Find best matching ground truth
        for (gt_idx, gt_box) in gt_boxes.iter().enumerate() {
            if matched_gt.contains(&gt_idx) {
                continue; // Already matched
            }

            let iou = calculate_iou(pred_box, gt_box);
            if iou > best_iou {
                best_iou = iou;
                best_gt_idx = Some(gt_idx);
            }
        }

        // Determine if this is a true positive
        let (is_tp, matched_gt_id) = if best_iou >= iou_threshold {
            if let Some(gt_idx) = best_gt_idx {
                matched_gt.insert(gt_idx);
                (true, Some(ground_truths[gt_idx].id))
            } else {
                (false, None)
            }
        } else {
            (false, None)
        };

        matches_map.insert(
            pred_idx,
            Match {
                prediction_id: pred.id,
                ground_truth_id: matched_gt_id,
                iou: best_iou,
                is_true_positive: is_tp,
                confidence: pred.confidence(),
            },
        );
    }

    // Return matches sorted by confidence (descending)
    let mut matches: Vec<Match> = pred_indices
        .iter()
        .map(|&idx| matches_map[&idx].clone())
        .collect();

    matches.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(matches)
}

/// Group annotations by image_id and category_id.
pub fn group_annotations(
    annotations: &[Annotation],
) -> HashMap<(u64, u64), Vec<Annotation>> {
    let mut groups: HashMap<(u64, u64), Vec<Annotation>> = HashMap::new();

    for annotation in annotations {
        let key = (annotation.image_id, annotation.category_id);
        groups.entry(key).or_insert_with(Vec::new).push(annotation.clone());
    }

    groups
}

/// Calculate precision and recall at different confidence thresholds.
///
/// # Arguments
///
/// * `matches` - All matches across all images/categories (sorted by confidence descending)
/// * `total_ground_truths` - Total number of ground truth annotations
/// * `confidence_thresholds` - Confidence thresholds to evaluate
///
/// # Returns
///
/// Returns vectors of (threshold, precision) and (threshold, recall) tuples.
pub fn calculate_pr_at_thresholds(
    matches: &[Match],
    total_ground_truths: usize,
    confidence_thresholds: &[f64],
) -> (Vec<(f64, f64)>, Vec<(f64, f64)>) {
    let mut precision_at_thresholds = Vec::new();
    let mut recall_at_thresholds = Vec::new();

    for &threshold in confidence_thresholds {
        let mut tp = 0;
        let mut fp = 0;

        // Count TP and FP for predictions above threshold
        for m in matches {
            if m.confidence >= threshold {
                if m.is_true_positive {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if total_ground_truths > 0 {
            tp as f64 / total_ground_truths as f64
        } else {
            0.0
        };

        precision_at_thresholds.push((threshold, precision));
        recall_at_thresholds.push((threshold, recall));
    }

    (precision_at_thresholds, recall_at_thresholds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_annotation(
        id: u64,
        image_id: u64,
        category_id: u64,
        bbox: Vec<f64>,
        score: Option<f64>,
    ) -> Annotation {
        Annotation {
            id,
            image_id,
            category_id,
            bbox,
            area: None,
            iscrowd: None,
            score,
        }
    }

    #[test]
    fn test_perfect_match() {
        let predictions = vec![create_test_annotation(
            1,
            1,
            1,
            vec![10.0, 10.0, 50.0, 50.0],
            Some(0.9),
        )];

        let ground_truths = vec![create_test_annotation(
            1,
            1,
            1,
            vec![10.0, 10.0, 50.0, 50.0],
            None,
        )];

        let matches = match_detections(&predictions, &ground_truths, 0.5).unwrap();
        assert_eq!(matches.len(), 1);
        assert!(matches[0].is_true_positive);
        assert!(matches[0].iou > 0.99);
    }

    #[test]
    fn test_no_match() {
        let predictions = vec![create_test_annotation(
            1,
            1,
            1,
            vec![10.0, 10.0, 50.0, 50.0],
            Some(0.9),
        )];

        let ground_truths = vec![create_test_annotation(
            1,
            1,
            1,
            vec![200.0, 200.0, 50.0, 50.0],
            None,
        )];

        let matches = match_detections(&predictions, &ground_truths, 0.5).unwrap();
        assert_eq!(matches.len(), 1);
        assert!(!matches[0].is_true_positive);
    }

    #[test]
    fn test_confidence_sorting() {
        let predictions = vec![
            create_test_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], Some(0.5)),
            create_test_annotation(2, 1, 1, vec![20.0, 20.0, 50.0, 50.0], Some(0.9)),
            create_test_annotation(3, 1, 1, vec![30.0, 30.0, 50.0, 50.0], Some(0.7)),
        ];

        let ground_truths = vec![create_test_annotation(
            1,
            1,
            1,
            vec![20.0, 20.0, 50.0, 50.0],
            None,
        )];

        let matches = match_detections(&predictions, &ground_truths, 0.5).unwrap();

        // Matches should be sorted by confidence descending
        assert_eq!(matches[0].prediction_id, 2); // 0.9
        assert_eq!(matches[1].prediction_id, 3); // 0.7
        assert_eq!(matches[2].prediction_id, 1); // 0.5

        // Only the highest confidence prediction should match
        assert!(matches[0].is_true_positive);
        assert!(!matches[1].is_true_positive);
        assert!(!matches[2].is_true_positive);
    }

    #[test]
    fn test_group_annotations() {
        let annotations = vec![
            create_test_annotation(1, 1, 1, vec![10.0, 10.0, 50.0, 50.0], None),
            create_test_annotation(2, 1, 1, vec![20.0, 20.0, 50.0, 50.0], None),
            create_test_annotation(3, 1, 2, vec![30.0, 30.0, 50.0, 50.0], None),
            create_test_annotation(4, 2, 1, vec![40.0, 40.0, 50.0, 50.0], None),
        ];

        let groups = group_annotations(&annotations);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[&(1, 1)].len(), 2);
        assert_eq!(groups[&(1, 2)].len(), 1);
        assert_eq!(groups[&(2, 1)].len(), 1);
    }
}
