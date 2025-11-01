//! Average Precision (AP) and mean Average Precision (mAP) calculation.

use crate::metrics::precision_recall::interpolate_precision;

/// Calculate Average Precision (AP) from precision-recall curve.
///
/// Uses the COCO-style 101-point interpolation method.
///
/// # Arguments
///
/// * `precisions` - Vector of precision values (sorted by recall)
/// * `recalls` - Vector of recall values (sorted)
///
/// # Returns
///
/// Returns the Average Precision value (0.0 to 1.0).
///
/// # Example
///
/// ```
/// use coco_eval::metrics::ap::calculate_ap;
///
/// let precisions = vec![1.0, 1.0, 0.67, 0.75, 0.6];
/// let recalls = vec![0.25, 0.5, 0.5, 0.75, 0.75];
/// let ap = calculate_ap(&precisions, &recalls);
/// assert!(ap >= 0.0 && ap <= 1.0);
/// ```
pub fn calculate_ap(precisions: &[f64], recalls: &[f64]) -> f64 {
    if precisions.is_empty() || recalls.is_empty() {
        return 0.0;
    }

    let interpolated = interpolate_precision(precisions, recalls);

    // Average over all 101 recall levels
    interpolated.iter().sum::<f64>() / interpolated.len() as f64
}

/// Calculate mean Average Precision (mAP) across multiple classes.
///
/// # Arguments
///
/// * `class_aps` - Vector of AP values for each class
///
/// # Returns
///
/// Returns the mean Average Precision (0.0 to 1.0).
///
/// # Example
///
/// ```
/// use coco_eval::metrics::ap::calculate_map;
///
/// let class_aps = vec![0.8, 0.9, 0.75, 0.85];
/// let map = calculate_map(&class_aps);
/// assert!((map - 0.825).abs() < 1e-10);
/// ```
pub fn calculate_map(class_aps: &[f64]) -> f64 {
    if class_aps.is_empty() {
        return 0.0;
    }

    class_aps.iter().sum::<f64>() / class_aps.len() as f64
}

/// Calculate AP at a specific IoU threshold.
///
/// This is a placeholder for the full implementation that would take
/// ground truth and predictions and compute AP at a specific IoU threshold.
///
/// # Arguments
///
/// * `precisions` - Precision values at different confidence thresholds
/// * `recalls` - Recall values at different confidence thresholds
/// * `_iou_threshold` - IoU threshold (e.g., 0.5 for AP50, 0.75 for AP75)
///
/// # Returns
///
/// Returns the AP at the specified IoU threshold.
pub fn calculate_ap_at_iou(
    precisions: &[f64],
    recalls: &[f64],
    _iou_threshold: f64,
) -> f64 {
    calculate_ap(precisions, recalls)
}

/// Calculate mAP across multiple IoU thresholds (COCO-style).
///
/// COCO mAP is the average of AP values computed at IoU thresholds
/// from 0.5 to 0.95 with a step size of 0.05.
///
/// # Arguments
///
/// * `ap_at_ious` - Vector of AP values at different IoU thresholds
///
/// # Returns
///
/// Returns the mean AP across IoU thresholds.
///
/// # Example
///
/// ```
/// use coco_eval::metrics::ap::calculate_coco_map;
///
/// // AP values at IoU thresholds [0.5, 0.55, 0.6, ..., 0.95]
/// let ap_values = vec![0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45];
/// let coco_map = calculate_coco_map(&ap_values);
/// assert!((coco_map - 0.675).abs() < 0.001);
/// ```
pub fn calculate_coco_map(ap_at_ious: &[f64]) -> f64 {
    calculate_map(ap_at_ious)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_ap_empty() {
        let ap = calculate_ap(&[], &[]);
        assert_eq!(ap, 0.0);
    }

    #[test]
    fn test_calculate_ap_perfect() {
        let precisions = vec![1.0; 10];
        let recalls = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        let ap = calculate_ap(&precisions, &recalls);
        assert!((ap - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_map() {
        let class_aps = vec![0.8, 0.9, 0.75, 0.85];
        let map = calculate_map(&class_aps);
        assert!((map - 0.825).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_map_empty() {
        let map = calculate_map(&[]);
        assert_eq!(map, 0.0);
    }

    #[test]
    fn test_calculate_coco_map() {
        let ap_values = vec![0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45];
        let coco_map = calculate_coco_map(&ap_values);
        assert!((coco_map - 0.675).abs() < 1e-10);
    }
}
