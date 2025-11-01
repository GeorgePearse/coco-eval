//! Precision and Recall calculation.

use crate::types::PrecisionRecallPoint;

/// Container for precision and recall values.
#[derive(Debug, Clone)]
pub struct PrecisionRecall {
    pub precision: f64,
    pub recall: f64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

/// Calculate precision and recall from TP, FP, and FN counts.
///
/// # Arguments
///
/// * `true_positives` - Number of true positive detections
/// * `false_positives` - Number of false positive detections
/// * `false_negatives` - Number of false negative (missed) detections
///
/// # Returns
///
/// Returns a `PrecisionRecall` struct containing precision and recall values.
///
/// # Example
///
/// ```
/// use coco_eval::metrics::precision_recall::calculate_precision_recall;
///
/// let pr = calculate_precision_recall(8, 2, 3);
/// assert_eq!(pr.precision, 0.8); // 8 / (8 + 2)
/// assert!((pr.recall - 0.7272).abs() < 0.001); // 8 / (8 + 3)
/// ```
pub fn calculate_precision_recall(
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
) -> PrecisionRecall {
    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };

    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else {
        0.0
    };

    PrecisionRecall {
        precision,
        recall,
        true_positives,
        false_positives,
        false_negatives,
    }
}

/// Calculate precision-recall curve from sorted detection scores.
///
/// # Arguments
///
/// * `is_true_positive` - Boolean array indicating if each detection is a true positive (sorted by confidence)
/// * `num_ground_truth` - Total number of ground truth annotations
///
/// # Returns
///
/// Returns a vector of precision-recall points.
pub fn calculate_precision_recall_curve(
    is_true_positive: &[bool],
    num_ground_truth: usize,
) -> Vec<PrecisionRecallPoint> {
    let mut curve = Vec::new();
    let mut tp = 0;
    let mut fp = 0;

    for (i, &is_tp) in is_true_positive.iter().enumerate() {
        if is_tp {
            tp += 1;
        } else {
            fp += 1;
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if num_ground_truth > 0 {
            tp as f64 / num_ground_truth as f64
        } else {
            0.0
        };

        curve.push(PrecisionRecallPoint {
            precision,
            recall,
            threshold: i as f64, // This would be replaced with actual confidence scores
        });
    }

    curve
}

/// Interpolate precision values for standard recall levels.
///
/// Uses the COCO-style 101-point interpolation.
///
/// # Arguments
///
/// * `precision` - Precision values (sorted by recall)
/// * `recall` - Recall values (sorted)
///
/// # Returns
///
/// Returns interpolated precision at 101 recall levels (0.0, 0.01, ..., 1.0).
pub fn interpolate_precision(precision: &[f64], recall: &[f64]) -> Vec<f64> {
    let recall_levels: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    let mut interpolated = vec![0.0; 101];

    for (i, &recall_level) in recall_levels.iter().enumerate() {
        // Find all precision values at recall >= recall_level
        let max_precision = precision
            .iter()
            .zip(recall.iter())
            .filter(|(_, &r)| r >= recall_level)
            .map(|(&p, _)| p)
            .fold(0.0f64, |a, b| a.max(b));

        interpolated[i] = max_precision;
    }

    interpolated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_precision_recall() {
        let pr = calculate_precision_recall(10, 0, 0);
        assert_eq!(pr.precision, 1.0);
        assert_eq!(pr.recall, 1.0);
    }

    #[test]
    fn test_zero_precision() {
        let pr = calculate_precision_recall(0, 10, 5);
        assert_eq!(pr.precision, 0.0);
        assert_eq!(pr.recall, 0.0);
    }

    #[test]
    fn test_precision_recall_values() {
        let pr = calculate_precision_recall(8, 2, 3);
        assert!((pr.precision - 0.8).abs() < 1e-10);
        assert!((pr.recall - 8.0 / 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall_curve() {
        let is_tp = vec![true, true, false, true, false];
        let curve = calculate_precision_recall_curve(&is_tp, 4);
        assert_eq!(curve.len(), 5);

        // First detection: TP
        assert!((curve[0].precision - 1.0).abs() < 1e-10);
        assert!((curve[0].recall - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_precision() {
        let precision = vec![1.0, 1.0, 0.67, 0.75, 0.6];
        let recall = vec![0.25, 0.5, 0.5, 0.75, 0.75];

        let interpolated = interpolate_precision(&precision, &recall);
        assert_eq!(interpolated.len(), 101);
    }
}
