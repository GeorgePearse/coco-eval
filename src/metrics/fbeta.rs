/// F-beta score calculation and threshold optimization
///
/// This module provides functions for calculating F-beta scores, precision, and recall
/// across different confidence thresholds for multi-class object detection evaluation.
use serde::{Deserialize, Serialize};

/// Result of F-beta evaluation at a specific threshold for a specific class
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FBetaResult {
    /// Class identifier
    pub class_id: i64,

    /// Confidence threshold used
    pub confidence_threshold: f64,

    /// Precision: TP / (TP + FP)
    pub precision: f64,

    /// Recall: TP / (TP + FN)
    pub recall: f64,

    /// F-beta score
    pub fbeta: f64,

    /// True positives count
    pub true_positives: usize,

    /// False positives count
    pub false_positives: usize,

    /// False negatives count
    pub false_negatives: usize,
}

/// Calculate precision from confusion matrix values
///
/// Precision = TP / (TP + FP)
///
/// # Arguments
///
/// * `tp` - True positives count
/// * `fp` - False positives count
///
/// # Returns
///
/// Precision value between 0.0 and 1.0
///
/// # Examples
///
/// ```
/// # use coco_eval::metrics::fbeta::calculate_precision;
/// let precision = calculate_precision(80, 20);
/// assert_eq!(precision, 0.8);
/// ```
#[must_use]
pub fn calculate_precision(tp: usize, fp: usize) -> f64 {
    let denominator = tp + fp;
    if denominator == 0 {
        return 0.0;
    }

    #[allow(clippy::cast_precision_loss)]
    let precision = (tp as f64) / (denominator as f64);

    // Ensure precision is in valid range [0.0, 1.0]
    debug_assert!(
        (0.0..=1.0).contains(&precision),
        "Precision must be between 0 and 1, got {precision}"
    );

    precision
}

/// Calculate recall from confusion matrix values
///
/// Recall = TP / (TP + FN)
///
/// # Arguments
///
/// * `tp` - True positives count
/// * `fn_` - False negatives count
///
/// # Returns
///
/// Recall value between 0.0 and 1.0
///
/// # Examples
///
/// ```
/// # use coco_eval::metrics::fbeta::calculate_recall;
/// let recall = calculate_recall(80, 20);
/// assert_eq!(recall, 0.8);
/// ```
#[must_use]
pub fn calculate_recall(tp: usize, fn_: usize) -> f64 {
    let denominator = tp + fn_;
    if denominator == 0 {
        return 0.0;
    }

    #[allow(clippy::cast_precision_loss)]
    let recall = (tp as f64) / (denominator as f64);

    // Ensure recall is in valid range [0.0, 1.0]
    debug_assert!(
        (0.0..=1.0).contains(&recall),
        "Recall must be between 0 and 1, got {recall}"
    );

    recall
}

/// Calculate F1 score from precision and recall
///
/// F1 = 2 * (precision * recall) / (precision + recall)
///
/// This is a special case of F-beta where beta = 1.0 (equal weight to precision and recall).
///
/// # Arguments
///
/// * `precision` - Precision value
/// * `recall` - Recall value
///
/// # Returns
///
/// F1 score value between 0.0 and 1.0
///
/// # Examples
///
/// ```
/// # use coco_eval::metrics::fbeta::calculate_f1;
/// let f1 = calculate_f1(0.8, 0.9);
/// assert!((f1 - 0.847).abs() < 0.001);
/// ```
#[must_use]
pub fn calculate_f1(precision: f64, recall: f64) -> f64 {
    calculate_fbeta(precision, recall, 1.0)
}

/// Calculate F-beta score from precision and recall
///
/// F-beta = (1 + beta²) * precision * recall / ((beta² * precision) + recall)
///
/// The beta parameter controls the trade-off between precision and recall:
/// - beta = 1.0: Equal weight (F1 score)
/// - beta > 1.0: More weight on recall
/// - beta < 1.0: More weight on precision
///
/// # Arguments
///
/// * `precision` - Precision value (0.0 to 1.0)
/// * `recall` - Recall value (0.0 to 1.0)
/// * `beta` - Beta parameter (must be positive)
///
/// # Returns
///
/// F-beta score value between 0.0 and 1.0
///
/// # Panics
///
/// Panics if beta is not positive
///
/// # Examples
///
/// ```
/// # use coco_eval::metrics::fbeta::calculate_fbeta;
/// // F1 score (beta = 1.0)
/// let f1 = calculate_fbeta(0.8, 0.9, 1.0);
/// assert!((f1 - 0.847).abs() < 0.001);
///
/// // F2 score (beta = 2.0, favors recall)
/// let f2 = calculate_fbeta(0.8, 0.9, 2.0);
/// assert!((f2 - 0.877).abs() < 0.001);
///
/// // F0.5 score (beta = 0.5, favors precision)
/// let f_half = calculate_fbeta(0.8, 0.9, 0.5);
/// assert!((f_half - 0.813).abs() < 0.001);
/// ```
#[must_use]
pub fn calculate_fbeta(precision: f64, recall: f64, beta: f64) -> f64 {
    assert!(beta > 0.0, "Beta must be positive, got {beta}");

    // Handle edge cases
    if precision + recall == 0.0 {
        return 0.0;
    }

    let beta_squared = beta * beta;
    let numerator = (1.0 + beta_squared) * precision * recall;
    let denominator = (beta_squared * precision) + recall;

    if denominator == 0.0 {
        return 0.0;
    }

    let fbeta = numerator / denominator;

    // Ensure F-beta is in valid range [0.0, 1.0]
    debug_assert!(
        (0.0..=1.0).contains(&fbeta),
        "F-beta must be between 0 and 1, got {fbeta} (precision={precision}, recall={recall}, beta={beta})"
    );

    fbeta
}

/// Calculate F-beta score directly from confusion matrix values
///
/// # Arguments
///
/// * `tp` - True positives count
/// * `fp` - False positives count
/// * `fn_` - False negatives count
/// * `beta` - Beta parameter (must be positive)
///
/// # Returns
///
/// F-beta score value between 0.0 and 1.0
///
/// # Panics
///
/// Panics if beta is not positive
///
/// # Examples
///
/// ```
/// # use coco_eval::metrics::fbeta::calculate_fbeta_from_counts;
/// let f1 = calculate_fbeta_from_counts(80, 20, 10, 1.0);
/// // precision = 80/(80+20) = 0.8
/// // recall = 80/(80+10) = 0.889
/// // f1 = 2 * 0.8 * 0.889 / (0.8 + 0.889) = 0.842
/// assert!((f1 - 0.842).abs() < 0.001);
/// ```
#[must_use]
pub fn calculate_fbeta_from_counts(tp: usize, fp: usize, fn_: usize, beta: f64) -> f64 {
    let precision = calculate_precision(tp, fp);
    let recall = calculate_recall(tp, fn_);
    calculate_fbeta(precision, recall, beta)
}

/// Generate a range of confidence thresholds for evaluation
///
/// Creates evenly spaced thresholds from 0.0 to 1.0 (inclusive).
///
/// # Arguments
///
/// * `num_thresholds` - Number of thresholds to generate (must be >= 2)
///
/// # Returns
///
/// Vector of threshold values
///
/// # Panics
///
/// Panics if `num_thresholds` < 2
///
/// # Examples
///
/// ```
/// # use coco_eval::metrics::fbeta::generate_threshold_range;
/// let thresholds = generate_threshold_range(5);
/// assert_eq!(thresholds, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
///
/// let thresholds = generate_threshold_range(11);
/// assert_eq!(thresholds.len(), 11);
/// assert_eq!(thresholds[0], 0.0);
/// assert_eq!(thresholds[10], 1.0);
/// ```
#[must_use]
pub fn generate_threshold_range(num_thresholds: usize) -> Vec<f64> {
    assert!(
        num_thresholds >= 2,
        "Number of thresholds must be at least 2, got {num_thresholds}"
    );

    (0..num_thresholds)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let step = i as f64 / ((num_thresholds - 1) as f64);
            step
        })
        .collect()
}

/// Find the threshold with the best F-beta score
///
/// # Arguments
///
/// * `results` - Slice of F-beta results for a single class
///
/// # Returns
///
/// Reference to the result with the highest F-beta score, or `None` if results are empty
///
/// # Examples
///
/// ```
/// # use coco_eval::metrics::fbeta::{FBetaResult, find_best_threshold};
/// let results = vec![
///     FBetaResult {
///         class_id: 1,
///         confidence_threshold: 0.3,
///         precision: 0.7,
///         recall: 0.9,
///         fbeta: 0.788,
///         true_positives: 90,
///         false_positives: 38,
///         false_negatives: 10,
///     },
///     FBetaResult {
///         class_id: 1,
///         confidence_threshold: 0.5,
///         precision: 0.85,
///         recall: 0.85,
///         fbeta: 0.85,
///         true_positives: 85,
///         false_positives: 15,
///         false_negatives: 15,
///     },
/// ];
///
/// let best = find_best_threshold(&results).unwrap();
/// assert_eq!(best.confidence_threshold, 0.5);
/// assert_eq!(best.fbeta, 0.85);
/// ```
#[must_use]
pub fn find_best_threshold(results: &[FBetaResult]) -> Option<&FBetaResult> {
    results
        .iter()
        .max_by(|a, b| a.fbeta.partial_cmp(&b.fbeta).unwrap_or(std::cmp::Ordering::Equal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_precision_perfect() {
        assert_eq!(calculate_precision(100, 0), 1.0);
    }

    #[test]
    fn test_calculate_precision_zero() {
        assert_eq!(calculate_precision(0, 100), 0.0);
    }

    #[test]
    fn test_calculate_precision_no_predictions() {
        assert_eq!(calculate_precision(0, 0), 0.0);
    }

    #[test]
    fn test_calculate_precision_typical() {
        assert_eq!(calculate_precision(80, 20), 0.8);
    }

    #[test]
    fn test_calculate_recall_perfect() {
        assert_eq!(calculate_recall(100, 0), 1.0);
    }

    #[test]
    fn test_calculate_recall_zero() {
        assert_eq!(calculate_recall(0, 100), 0.0);
    }

    #[test]
    fn test_calculate_recall_no_ground_truth() {
        assert_eq!(calculate_recall(0, 0), 0.0);
    }

    #[test]
    fn test_calculate_recall_typical() {
        let recall = calculate_recall(80, 20);
        assert_eq!(recall, 0.8);
    }

    #[test]
    fn test_calculate_f1_equal_precision_recall() {
        let f1 = calculate_f1(0.8, 0.8);
        assert!((f1 - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_f1_different_values() {
        let f1 = calculate_f1(0.8, 0.9);
        assert!((f1 - 0.847058).abs() < 0.001);
    }

    #[test]
    fn test_calculate_f1_zero_precision() {
        assert_eq!(calculate_f1(0.0, 0.9), 0.0);
    }

    #[test]
    fn test_calculate_f1_zero_recall() {
        assert_eq!(calculate_f1(0.9, 0.0), 0.0);
    }

    #[test]
    fn test_calculate_fbeta_f1() {
        // beta=1.0 should give same result as calculate_f1
        let f1 = calculate_f1(0.8, 0.9);
        let fbeta = calculate_fbeta(0.8, 0.9, 1.0);
        assert!((f1 - fbeta).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_fbeta_f2_favors_recall() {
        // F2 score should be closer to recall than precision
        let precision = 0.7;
        let recall = 0.9;
        let f2 = calculate_fbeta(precision, recall, 2.0);

        // F2 should be between precision and recall, closer to recall
        assert!(f2 > precision);
        assert!(f2 < recall);
        assert!((f2 - recall).abs() < (f2 - precision).abs());
    }

    #[test]
    fn test_calculate_fbeta_f_half_favors_precision() {
        // F0.5 score should be closer to precision than recall
        let precision = 0.9;
        let recall = 0.7;
        let f_half = calculate_fbeta(precision, recall, 0.5);

        // F0.5 should be between precision and recall, closer to precision
        assert!(f_half > recall);
        assert!(f_half < precision);
        assert!((f_half - precision).abs() < (f_half - recall).abs());
    }

    #[test]
    #[should_panic(expected = "Beta must be positive")]
    fn test_calculate_fbeta_negative_beta() {
        let _ = calculate_fbeta(0.8, 0.9, -1.0);
    }

    #[test]
    #[should_panic(expected = "Beta must be positive")]
    fn test_calculate_fbeta_zero_beta() {
        let _ = calculate_fbeta(0.8, 0.9, 0.0);
    }

    #[test]
    fn test_calculate_fbeta_from_counts() {
        let f1 = calculate_fbeta_from_counts(80, 20, 10, 1.0);

        // precision = 80/(80+20) = 0.8
        // recall = 80/(80+10) = 0.888...
        // f1 = 2 * 0.8 * 0.888... / (0.8 + 0.888...) ≈ 0.8421
        assert!((f1 - 0.8421).abs() < 0.001);
    }

    #[test]
    fn test_generate_threshold_range_simple() {
        let thresholds = generate_threshold_range(5);
        assert_eq!(thresholds, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_generate_threshold_range_eleven() {
        let thresholds = generate_threshold_range(11);
        assert_eq!(thresholds.len(), 11);
        assert_eq!(thresholds[0], 0.0);
        assert_eq!(thresholds[10], 1.0);
        assert_eq!(thresholds[5], 0.5);
    }

    #[test]
    fn test_generate_threshold_range_coco_default() {
        // COCO typically uses 101 thresholds
        let thresholds = generate_threshold_range(101);
        assert_eq!(thresholds.len(), 101);
        assert_eq!(thresholds[0], 0.0);
        assert_eq!(thresholds[100], 1.0);
        assert!((thresholds[50] - 0.5).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Number of thresholds must be at least 2")]
    fn test_generate_threshold_range_too_few() {
        let _ = generate_threshold_range(1);
    }

    #[test]
    fn test_find_best_threshold() {
        let results = vec![
            FBetaResult {
                class_id: 1,
                confidence_threshold: 0.3,
                precision: 0.7,
                recall: 0.9,
                fbeta: 0.788,
                true_positives: 90,
                false_positives: 38,
                false_negatives: 10,
            },
            FBetaResult {
                class_id: 1,
                confidence_threshold: 0.5,
                precision: 0.85,
                recall: 0.85,
                fbeta: 0.85,
                true_positives: 85,
                false_positives: 15,
                false_negatives: 15,
            },
            FBetaResult {
                class_id: 1,
                confidence_threshold: 0.7,
                precision: 0.95,
                recall: 0.70,
                fbeta: 0.806,
                true_positives: 70,
                false_positives: 3,
                false_negatives: 30,
            },
        ];

        let best = find_best_threshold(&results).unwrap();
        assert_eq!(best.confidence_threshold, 0.5);
        assert_eq!(best.fbeta, 0.85);
    }

    #[test]
    fn test_find_best_threshold_empty() {
        let results: Vec<FBetaResult> = vec![];
        assert!(find_best_threshold(&results).is_none());
    }

    #[test]
    fn test_find_best_threshold_single() {
        let results = vec![FBetaResult {
            class_id: 1,
            confidence_threshold: 0.5,
            precision: 0.8,
            recall: 0.9,
            fbeta: 0.847,
            true_positives: 90,
            false_positives: 22,
            false_negatives: 10,
        }];

        let best = find_best_threshold(&results).unwrap();
        assert_eq!(best.confidence_threshold, 0.5);
    }

    #[test]
    fn test_fbeta_result_creation() {
        let result = FBetaResult {
            class_id: 42,
            confidence_threshold: 0.75,
            precision: 0.9,
            recall: 0.8,
            fbeta: 0.847,
            true_positives: 80,
            false_positives: 8,
            false_negatives: 20,
        };

        assert_eq!(result.class_id, 42);
        assert_eq!(result.confidence_threshold, 0.75);
        assert_eq!(result.precision, 0.9);
        assert_eq!(result.recall, 0.8);
    }
}
