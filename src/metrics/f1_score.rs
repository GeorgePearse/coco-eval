//! F1 Score calculation.

use crate::metrics::precision_recall::PrecisionRecall;

/// Calculate F1 score from precision and recall.
///
/// F1 score is the harmonic mean of precision and recall:
/// F1 = 2 × (Precision × Recall) / (Precision + Recall)
///
/// # Arguments
///
/// * `precision` - Precision value (0.0 to 1.0)
/// * `recall` - Recall value (0.0 to 1.0)
///
/// # Returns
///
/// Returns the F1 score (0.0 to 1.0). Returns 0.0 if both precision and recall are 0.
///
/// # Example
///
/// ```
/// use coco_eval::metrics::f1_score::calculate_f1_score;
///
/// let f1 = calculate_f1_score(0.8, 0.6);
/// assert!((f1 - 0.6857).abs() < 0.001);
/// ```
pub fn calculate_f1_score(precision: f64, recall: f64) -> f64 {
    if precision + recall == 0.0 {
        return 0.0;
    }

    2.0 * (precision * recall) / (precision + recall)
}

/// Calculate F1 score from a PrecisionRecall struct.
///
/// # Arguments
///
/// * `pr` - PrecisionRecall struct containing precision and recall values
///
/// # Returns
///
/// Returns the F1 score (0.0 to 1.0).
///
/// # Example
///
/// ```
/// use coco_eval::metrics::precision_recall::calculate_precision_recall;
/// use coco_eval::metrics::f1_score::calculate_f1_from_pr;
///
/// let pr = calculate_precision_recall(8, 2, 3);
/// let f1 = calculate_f1_from_pr(&pr);
/// assert!(f1 > 0.0 && f1 <= 1.0);
/// ```
pub fn calculate_f1_from_pr(pr: &PrecisionRecall) -> f64 {
    calculate_f1_score(pr.precision, pr.recall)
}

/// Calculate F1 score directly from TP, FP, and FN counts.
///
/// # Arguments
///
/// * `true_positives` - Number of true positive detections
/// * `false_positives` - Number of false positive detections
/// * `false_negatives` - Number of false negative detections
///
/// # Returns
///
/// Returns the F1 score (0.0 to 1.0).
///
/// # Example
///
/// ```
/// use coco_eval::metrics::f1_score::calculate_f1_from_counts;
///
/// let f1 = calculate_f1_from_counts(8, 2, 3);
/// assert!((f1 - 0.7273).abs() < 0.001);
/// ```
pub fn calculate_f1_from_counts(
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
) -> f64 {
    use crate::metrics::precision_recall::calculate_precision_recall;

    let pr = calculate_precision_recall(true_positives, false_positives, false_negatives);
    calculate_f1_from_pr(&pr)
}

/// Find the optimal F1 score and corresponding threshold from a precision-recall curve.
///
/// # Arguments
///
/// * `precisions` - Vector of precision values
/// * `recalls` - Vector of recall values
/// * `thresholds` - Vector of confidence thresholds
///
/// # Returns
///
/// Returns a tuple of (optimal_f1, optimal_threshold).
///
/// # Example
///
/// ```
/// use coco_eval::metrics::f1_score::find_optimal_f1_threshold;
///
/// let precisions = vec![1.0, 0.9, 0.8, 0.7];
/// let recalls = vec![0.5, 0.6, 0.7, 0.8];
/// let thresholds = vec![0.9, 0.8, 0.7, 0.6];
///
/// let (f1, threshold) = find_optimal_f1_threshold(&precisions, &recalls, &thresholds);
/// assert!(f1 > 0.0);
/// ```
pub fn find_optimal_f1_threshold(
    precisions: &[f64],
    recalls: &[f64],
    thresholds: &[f64],
) -> (f64, f64) {
    let mut best_f1 = 0.0;
    let mut best_threshold = 0.0;

    for i in 0..precisions.len().min(recalls.len()).min(thresholds.len()) {
        let f1 = calculate_f1_score(precisions[i], recalls[i]);
        if f1 > best_f1 {
            best_f1 = f1;
            best_threshold = thresholds[i];
        }
    }

    (best_f1, best_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_f1() {
        let f1 = calculate_f1_score(1.0, 1.0);
        assert!((f1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_f1() {
        let f1 = calculate_f1_score(0.0, 0.0);
        assert_eq!(f1, 0.0);
    }

    #[test]
    fn test_f1_calculation() {
        let f1 = calculate_f1_score(0.8, 0.6);
        // F1 = 2 * (0.8 * 0.6) / (0.8 + 0.6) = 0.96 / 1.4 ≈ 0.6857
        assert!((f1 - 0.685714).abs() < 1e-5);
    }

    #[test]
    fn test_f1_from_counts() {
        let f1 = calculate_f1_from_counts(8, 2, 3);
        // Precision = 8/10 = 0.8
        // Recall = 8/11 ≈ 0.7273
        // F1 = 2 * (0.8 * 0.7273) / (0.8 + 0.7273) ≈ 0.7619
        assert!((f1 - 0.7619).abs() < 0.001);
    }

    #[test]
    fn test_find_optimal_f1() {
        let precisions = vec![1.0, 0.9, 0.8, 0.7];
        let recalls = vec![0.5, 0.6, 0.7, 0.8];
        let thresholds = vec![0.9, 0.8, 0.7, 0.6];

        let (f1, threshold) = find_optimal_f1_threshold(&precisions, &recalls, &thresholds);
        assert!(f1 > 0.0);
        assert!(threshold >= 0.6 && threshold <= 0.9);
    }
}
