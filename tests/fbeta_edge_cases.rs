//! Edge case tests for F-beta metrics module.
//!
//! This test suite focuses on boundary conditions, edge cases, and special scenarios
//! for precision, recall, F1, and F-beta calculations.

use coco_eval::metrics::{
    calculate_fbeta, calculate_f1, calculate_precision, calculate_recall,
    find_best_threshold, generate_threshold_range, FBetaResult,
};

#[test]
fn test_precision_with_zero_predictions() {
    // When there are no predictions (tp=0, fp=0), precision should be 0
    let precision = calculate_precision(0, 0);
    assert_eq!(precision, 0.0);
}

#[test]
fn test_recall_with_zero_ground_truth() {
    // When there is no ground truth (tp=0, fn=0), recall should be 0
    let recall = calculate_recall(0, 0);
    assert_eq!(recall, 0.0);
}

#[test]
fn test_precision_perfect_predictions() {
    // All predictions are correct (no false positives)
    let precision = calculate_precision(100, 0);
    assert_eq!(precision, 1.0);
}

#[test]
fn test_recall_perfect_coverage() {
    // All ground truth detected (no false negatives)
    let recall = calculate_recall(100, 0);
    assert_eq!(recall, 1.0);
}

#[test]
fn test_precision_all_false_positives() {
    // No true positives, only false positives
    let precision = calculate_precision(0, 50);
    assert_eq!(precision, 0.0);
}

#[test]
fn test_recall_all_false_negatives() {
    // No true positives, only false negatives
    let recall = calculate_recall(0, 50);
    assert_eq!(recall, 0.0);
}

#[test]
fn test_f1_both_zero() {
    // When both precision and recall are 0, F1 should be 0
    let f1 = calculate_f1(0.0, 0.0);
    assert_eq!(f1, 0.0);
}

#[test]
fn test_f1_one_zero() {
    // When either precision or recall is 0, F1 should be 0
    assert_eq!(calculate_f1(0.0, 0.5), 0.0);
    assert_eq!(calculate_f1(0.5, 0.0), 0.0);
}

#[test]
fn test_f1_perfect_scores() {
    // Perfect precision and recall should give F1 = 1.0
    let f1 = calculate_f1(1.0, 1.0);
    assert!((f1 - 1.0).abs() < 1e-10);
}

#[test]
fn test_f1_harmonic_mean() {
    // F1 should be the harmonic mean of precision and recall
    let precision = 0.8;
    let recall = 0.6;
    let f1 = calculate_f1(precision, recall);
    let expected = 2.0 * precision * recall / (precision + recall);
    assert!((f1 - expected).abs() < 1e-10);
}

#[test]
fn test_fbeta_with_beta_zero() {
    // Beta = 0 should give only precision (recall ignored)
    // F_beta = (1 + beta^2) * (P * R) / (beta^2 * P + R)
    // When beta -> 0, this approaches precision
    let precision = 0.8;
    let recall = 0.3;
    let fbeta = calculate_fbeta(precision, recall, 0.001);
    assert!((fbeta - precision).abs() < 0.01);
}

#[test]
fn test_fbeta_with_very_large_beta() {
    // Very large beta should approach recall
    let precision = 0.8;
    let recall = 0.3;
    let fbeta = calculate_fbeta(precision, recall, 100.0);
    assert!((fbeta - recall).abs() < 0.01);
}

#[test]
fn test_fbeta_beta_one_equals_f1() {
    // F-beta with beta=1 should equal F1
    let precision = 0.75;
    let recall = 0.65;
    let f1 = calculate_f1(precision, recall);
    let fbeta = calculate_fbeta(precision, recall, 1.0);
    assert!((f1 - fbeta).abs() < 1e-10);
}

#[test]
fn test_fbeta_beta_two() {
    // F2 should favor recall more than F1
    let precision = 0.9;
    let recall = 0.5;
    let f1 = calculate_fbeta(precision, recall, 1.0);
    let f2 = calculate_fbeta(precision, recall, 2.0);
    // F2 should be lower than F1 when precision > recall
    assert!(f2 < f1);
}

#[test]
fn test_fbeta_beta_half() {
    // F0.5 should favor precision more than F1
    let precision = 0.5;
    let recall = 0.9;
    let f1 = calculate_fbeta(precision, recall, 1.0);
    let f_half = calculate_fbeta(precision, recall, 0.5);
    // F0.5 should be lower than F1 when recall > precision
    assert!(f_half < f1);
}

#[test]
fn test_fbeta_both_zero() {
    // When both precision and recall are 0, F-beta should be 0
    let fbeta = calculate_fbeta(0.0, 0.0, 1.5);
    assert_eq!(fbeta, 0.0);
}

#[test]
fn test_fbeta_precision_zero() {
    // When precision is 0, F-beta should be 0 regardless of recall
    let fbeta = calculate_fbeta(0.0, 0.8, 1.5);
    assert_eq!(fbeta, 0.0);
}

#[test]
fn test_fbeta_recall_zero() {
    // When recall is 0, F-beta should be 0 regardless of precision
    let fbeta = calculate_fbeta(0.8, 0.0, 1.5);
    assert_eq!(fbeta, 0.0);
}

#[test]
#[should_panic(expected = "Number of thresholds must be at least 2")]
fn test_threshold_range_single() {
    // Single threshold should panic - minimum is 2
    generate_threshold_range(1);
}

#[test]
fn test_threshold_range_two() {
    // Two thresholds should be 0.0 and 1.0
    let thresholds = generate_threshold_range(2);
    assert_eq!(thresholds.len(), 2);
    assert!((thresholds[0] - 0.0).abs() < 1e-10);
    assert!((thresholds[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_threshold_range_eleven() {
    // Eleven thresholds should be 0.0, 0.1, 0.2, ..., 1.0
    let thresholds = generate_threshold_range(11);
    assert_eq!(thresholds.len(), 11);
    for (i, &threshold) in thresholds.iter().enumerate() {
        let expected = i as f64 / 10.0;
        assert!((threshold - expected).abs() < 1e-10);
    }
}

#[test]
fn test_threshold_range_monotonic() {
    // Thresholds should be monotonically increasing
    let thresholds = generate_threshold_range(50);
    for i in 1..thresholds.len() {
        assert!(thresholds[i] > thresholds[i - 1]);
    }
}

#[test]
fn test_threshold_range_bounds() {
    // First threshold should be 0.0, last should be 1.0
    let thresholds = generate_threshold_range(100);
    assert!((thresholds[0] - 0.0).abs() < 1e-10);
    assert!((thresholds[thresholds.len() - 1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_find_best_threshold_empty() {
    // Empty results should return None
    let results: Vec<FBetaResult> = vec![];
    assert!(find_best_threshold(&results).is_none());
}

#[test]
fn test_find_best_threshold_single() {
    // Single result should be returned as best
    let results = vec![FBetaResult {
        class_id: 1,
        confidence_threshold: 0.5,
        precision: 0.8,
        recall: 0.7,
        fbeta: 0.75,
        true_positives: 7,
        false_positives: 2,
        false_negatives: 3,
    }];
    let best = find_best_threshold(&results).unwrap();
    assert_eq!(best.confidence_threshold, 0.5);
}

#[test]
fn test_find_best_threshold_multiple() {
    // Should find the result with highest F-beta
    let results = vec![
        FBetaResult {
            class_id: 1,
            confidence_threshold: 0.3,
            precision: 0.6,
            recall: 0.8,
            fbeta: 0.70,
            true_positives: 8,
            false_positives: 5,
            false_negatives: 2,
        },
        FBetaResult {
            class_id: 1,
            confidence_threshold: 0.5,
            precision: 0.8,
            recall: 0.7,
            fbeta: 0.85, // Highest
            true_positives: 7,
            false_positives: 2,
            false_negatives: 3,
        },
        FBetaResult {
            class_id: 1,
            confidence_threshold: 0.7,
            precision: 0.9,
            recall: 0.5,
            fbeta: 0.60,
            true_positives: 5,
            false_positives: 1,
            false_negatives: 5,
        },
    ];
    let best = find_best_threshold(&results).unwrap();
    assert_eq!(best.confidence_threshold, 0.5);
    assert!((best.fbeta - 0.85).abs() < 1e-10);
}

#[test]
fn test_find_best_threshold_ties() {
    // When there are ties, max_by returns the last one
    let results = vec![
        FBetaResult {
            class_id: 1,
            confidence_threshold: 0.3,
            precision: 0.7,
            recall: 0.7,
            fbeta: 0.8,
            true_positives: 7,
            false_positives: 3,
            false_negatives: 3,
        },
        FBetaResult {
            class_id: 1,
            confidence_threshold: 0.5,
            precision: 0.75,
            recall: 0.75,
            fbeta: 0.8, // Tie
            true_positives: 6,
            false_positives: 2,
            false_negatives: 2,
        },
    ];
    let best = find_best_threshold(&results).unwrap();
    // max_by returns the last maximum element when there are ties
    assert_eq!(best.confidence_threshold, 0.5);
}

#[test]
fn test_precision_large_numbers() {
    // Test with large numbers to check for overflow/precision issues
    let precision = calculate_precision(1_000_000, 500_000);
    assert!((precision - 0.666667).abs() < 1e-5);
}

#[test]
fn test_recall_large_numbers() {
    // Test with large numbers to check for overflow/precision issues
    let recall = calculate_recall(1_000_000, 500_000);
    assert!((recall - 0.666667).abs() < 1e-5);
}

#[test]
#[should_panic(expected = "Beta must be positive")]
fn test_fbeta_negative_beta() {
    // Negative beta doesn't make mathematical sense and should panic
    calculate_fbeta(0.8, 0.6, -1.0);
}

#[test]
fn test_precision_recall_extreme_ratios() {
    // Very high precision, low recall
    let precision = calculate_precision(1, 0);
    let recall = calculate_recall(1, 1000);
    assert_eq!(precision, 1.0);
    // 1 / 1001 ≈ 0.000999, not exactly 0.001
    assert!((recall - 1.0/1001.0).abs() < 1e-10);

    // Very low precision, high recall
    let precision2 = calculate_precision(1, 1000);
    let recall2 = calculate_recall(1, 0);
    // 1 / 1001 ≈ 0.000999, not exactly 0.001
    assert!((precision2 - 1.0/1001.0).abs() < 1e-10);
    assert_eq!(recall2, 1.0);
}

#[test]
fn test_f1_numerical_stability() {
    // Test F1 calculation with very small values
    let f1_small = calculate_f1(1e-10, 1e-10);
    assert!(f1_small >= 0.0 && f1_small <= 1.0);

    // Test with values very close to 1
    let f1_large = calculate_f1(0.9999999, 0.9999999);
    assert!((f1_large - 0.9999999).abs() < 1e-6);
}

#[test]
fn test_fbeta_common_beta_values() {
    // Test common beta values used in practice
    let precision = 0.7;
    let recall = 0.6;

    let f05 = calculate_fbeta(precision, recall, 0.5); // Favors precision
    let f1 = calculate_fbeta(precision, recall, 1.0); // Balanced
    let f2 = calculate_fbeta(precision, recall, 2.0); // Favors recall

    // Since precision > recall, we expect F0.5 > F1 > F2
    assert!(f05 > f1);
    assert!(f1 > f2);
}
