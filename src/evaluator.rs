//! Main evaluation orchestrator for COCO object detection metrics.

use crate::error::{CocoEvalError, Result};
use crate::matching::{calculate_pr_at_thresholds, group_annotations, match_detections, Match};
use crate::metrics::ap::{calculate_ap, calculate_map};
use crate::metrics::f1_score::calculate_f1_score;
use crate::types::{CocoDataset, EvaluationMetrics};
use std::collections::{HashMap, HashSet};

/// Evaluate object detection predictions against ground truth.
///
/// **Independent per-class evaluation**: This function performs an independent sweep
/// for each class, matching predictions to ground truth and computing metrics separately
/// per class before aggregating.
///
/// Computes mAP, AP50, AP75, precision, recall, and F1 scores.
///
/// # Arguments
///
/// * `ground_truth` - Dataset containing ground truth annotations
/// * `predictions` - Dataset containing predicted annotations with scores
/// * `iou_thresholds` - IoU thresholds to use for evaluation (default: 0.5:0.05:0.95)
/// * `confidence_thresholds` - Confidence thresholds for precision/recall curves
///
/// # Returns
///
/// Returns `EvaluationMetrics` containing all computed metrics.
pub fn evaluate(
    ground_truth: &CocoDataset,
    predictions: &CocoDataset,
    iou_thresholds: Option<Vec<f64>>,
    confidence_thresholds: Option<Vec<f64>>,
) -> Result<EvaluationMetrics> {
    let iou_thresholds = iou_thresholds.unwrap_or_else(|| {
        // Default COCO evaluation thresholds: 0.5:0.05:0.95
        (0..10).map(|i| 0.5 + 0.05 * i as f64).collect()
    });

    let confidence_thresholds = confidence_thresholds.unwrap_or_else(|| {
        // Default confidence thresholds: 0.0:0.1:1.0
        (0..11).map(|i| i as f64 * 0.1).collect()
    });

    // Get all unique category IDs
    let mut category_ids: HashSet<u64> = HashSet::new();
    for ann in &ground_truth.annotations {
        category_ids.insert(ann.category_id);
    }
    for ann in &predictions.annotations {
        category_ids.insert(ann.category_id);
    }

    if category_ids.is_empty() {
        return Err(CocoEvalError::EmptyDataset(
            "No categories found in datasets".to_string(),
        ));
    }

    let mut category_ids: Vec<u64> = category_ids.into_iter().collect();
    category_ids.sort_unstable();

    // Group annotations by (image_id, category_id)
    let gt_groups = group_annotations(&ground_truth.annotations);
    let pred_groups = group_annotations(&predictions.annotations);

    // Store AP values for each class and IoU threshold
    let mut ap_per_class_iou: HashMap<u64, Vec<f64>> = HashMap::new();

    // INDEPENDENT SWEEP FOR EACH CLASS
    for &category_id in &category_ids {
        let mut ap_values_for_class = Vec::new();

        // For each IoU threshold
        for &iou_threshold in &iou_thresholds {
            let ap = evaluate_single_class(
                &gt_groups,
                &pred_groups,
                category_id,
                iou_threshold,
            )?;
            ap_values_for_class.push(ap);
        }

        ap_per_class_iou.insert(category_id, ap_values_for_class);
    }

    // Aggregate metrics
    let mut metrics = EvaluationMetrics::new();

    // Calculate mAP (average across all classes and all IoU thresholds)
    let all_ap_values: Vec<f64> = ap_per_class_iou
        .values()
        .flat_map(|aps| aps.iter().copied())
        .collect();
    metrics.map = calculate_map(&all_ap_values);

    // Calculate AP50 (average across all classes at IoU=0.5)
    let iou_50_idx = iou_thresholds
        .iter()
        .position(|&x| (x - 0.5).abs() < 1e-6)
        .unwrap_or(0);
    let ap50_values: Vec<f64> = ap_per_class_iou
        .values()
        .filter_map(|aps| aps.get(iou_50_idx).copied())
        .collect();
    metrics.ap50 = calculate_map(&ap50_values);

    // Calculate AP75 (average across all classes at IoU=0.75)
    let iou_75_idx = iou_thresholds
        .iter()
        .position(|&x| (x - 0.75).abs() < 1e-6)
        .unwrap_or(iou_thresholds.len() - 1);
    let ap75_values: Vec<f64> = ap_per_class_iou
        .values()
        .filter_map(|aps| aps.get(iou_75_idx).copied())
        .collect();
    metrics.ap75 = calculate_map(&ap75_values);

    // Per-class AP (averaged across IoU thresholds)
    for (&category_id, ap_values) in &ap_per_class_iou {
        let class_ap = calculate_map(ap_values);
        metrics.ap_per_class.push((category_id, class_ap));
    }
    metrics.ap_per_class.sort_by_key(|&(id, _)| id);

    // Calculate precision/recall/F1 at confidence thresholds
    // Use IoU=0.5 for this calculation
    let (precision_at_thresh, recall_at_thresh) =
        calculate_metrics_at_confidence_thresholds(
            &gt_groups,
            &pred_groups,
            &category_ids,
            0.5,
            &confidence_thresholds,
        )?;

    metrics.precision_at_thresholds.clone_from(&precision_at_thresh);
    metrics.recall_at_thresholds.clone_from(&recall_at_thresh);

    // Calculate F1 scores
    for (i, _) in confidence_thresholds.iter().enumerate() {
        let precision = precision_at_thresh.get(i).map_or(0.0, |(_, p)| *p);
        let recall = recall_at_thresh.get(i).map_or(0.0, |(_, r)| *r);
        let f1 = calculate_f1_score(precision, recall);
        metrics
            .f1_at_thresholds
            .push((confidence_thresholds[i], f1));
    }

    Ok(metrics)
}

/// Evaluate a single class at a specific IoU threshold.
///
/// This is an independent sweep for one class.
fn evaluate_single_class(
    gt_groups: &HashMap<(u64, u64), Vec<crate::types::Annotation>>,
    pred_groups: &HashMap<(u64, u64), Vec<crate::types::Annotation>>,
    category_id: u64,
    iou_threshold: f64,
) -> Result<f64> {
    // Collect all matches for this class across all images
    let mut all_matches: Vec<Match> = Vec::new();
    let mut total_gt_count = 0;

    // Get all unique image IDs for this class
    let mut image_ids: HashSet<u64> = HashSet::new();
    for &(img_id, cat_id) in gt_groups.keys() {
        if cat_id == category_id {
            image_ids.insert(img_id);
        }
    }
    for &(img_id, cat_id) in pred_groups.keys() {
        if cat_id == category_id {
            image_ids.insert(img_id);
        }
    }

    // For each image, match predictions to ground truth
    for image_id in image_ids {
        let key = (image_id, category_id);

        let gts = gt_groups.get(&key).cloned().unwrap_or_default();
        let preds = pred_groups.get(&key).cloned().unwrap_or_default();

        total_gt_count += gts.len();

        if !preds.is_empty() {
            let matches = match_detections(&preds, &gts, iou_threshold)?;
            all_matches.extend(matches);
        }
    }

    // Sort all matches by confidence descending
    all_matches.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build precision-recall curve
    let (precisions, recalls) = build_precision_recall_curve(&all_matches, total_gt_count);

    // Calculate AP using COCO-style interpolation
    let ap = calculate_ap(&precisions, &recalls);

    Ok(ap)
}

/// Build precision-recall curve from matches.
fn build_precision_recall_curve(matches: &[Match], total_ground_truths: usize) -> (Vec<f64>, Vec<f64>) {
    let mut precisions = Vec::new();
    let mut recalls = Vec::new();

    let mut tp = 0;
    let mut fp = 0;

    for m in matches {
        if m.is_true_positive {
            tp += 1;
        } else {
            fp += 1;
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

        precisions.push(precision);
        recalls.push(recall);
    }

    (precisions, recalls)
}

/// Calculate precision and recall at different confidence thresholds.
fn calculate_metrics_at_confidence_thresholds(
    gt_groups: &HashMap<(u64, u64), Vec<crate::types::Annotation>>,
    pred_groups: &HashMap<(u64, u64), Vec<crate::types::Annotation>>,
    category_ids: &[u64],
    iou_threshold: f64,
    confidence_thresholds: &[f64],
) -> Result<(Vec<(f64, f64)>, Vec<(f64, f64)>)> {
    // Collect all matches across all classes and images
    let mut all_matches: Vec<Match> = Vec::new();
    let mut total_gt_count = 0;

    for &category_id in category_ids {
        let mut image_ids: HashSet<u64> = HashSet::new();
        for &(img_id, cat_id) in gt_groups.keys() {
            if cat_id == category_id {
                image_ids.insert(img_id);
            }
        }
        for &(img_id, cat_id) in pred_groups.keys() {
            if cat_id == category_id {
                image_ids.insert(img_id);
            }
        }

        for image_id in image_ids {
            let key = (image_id, category_id);
            let gts = gt_groups.get(&key).cloned().unwrap_or_default();
            let preds = pred_groups.get(&key).cloned().unwrap_or_default();

            total_gt_count += gts.len();

            if !preds.is_empty() {
                let matches = match_detections(&preds, &gts, iou_threshold)?;
                all_matches.extend(matches);
            }
        }
    }

    // Calculate precision/recall at each threshold
    let (precision_at_thresh, recall_at_thresh) =
        calculate_pr_at_thresholds(&all_matches, total_gt_count, confidence_thresholds);

    Ok((precision_at_thresh, recall_at_thresh))
}

/// Evaluate at a specific IoU threshold.
///
/// # Arguments
///
/// * `ground_truth` - Dataset containing ground truth annotations
/// * `predictions` - Dataset containing predicted annotations with scores
/// * `iou_threshold` - IoU threshold for matching (e.g., 0.5, 0.75)
///
/// # Returns
///
/// Returns average precision at the specified IoU threshold across all classes.
pub fn evaluate_at_iou(
    ground_truth: &CocoDataset,
    predictions: &CocoDataset,
    iou_threshold: f64,
) -> Result<f64> {
    let metrics = evaluate(ground_truth, predictions, Some(vec![iou_threshold]), None)?;
    Ok(metrics.map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Annotation, Category};

    fn create_test_dataset() -> CocoDataset {
        CocoDataset {
            images: None,
            annotations: vec![
                Annotation {
                    id: 1,
                    image_id: 1,
                    category_id: 1,
                    bbox: vec![10.0, 10.0, 50.0, 50.0],
                    area: Some(2500.0),
                    iscrowd: None,
                    score: Some(0.9),
                },
            ],
            categories: vec![
                Category {
                    id: 1,
                    name: "person".to_string(),
                    supercategory: None,
                },
            ],
        }
    }

    #[test]
    fn test_evaluate_basic() {
        let dataset = create_test_dataset();
        let result = evaluate(&dataset, &dataset, None, None);
        assert!(result.is_ok());
    }
}
