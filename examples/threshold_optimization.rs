//! Example demonstrating confidence threshold optimization for best F1 score.

use coco_eval::{
    generate_threshold_range, metrics::f1_score::find_optimal_f1_threshold,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Confidence Threshold Optimization Example ===\n");

    // Example 1: Generate threshold range
    println!("1. Generating Threshold Range");
    let thresholds = generate_threshold_range(0.0, 1.0, 11)?;
    println!("   Generated {} thresholds:", thresholds.len());
    println!("   {:?}", thresholds);
    println!();

    // Example 2: Simulate precision/recall at different thresholds
    println!("2. Simulated Precision/Recall at Different Thresholds");
    println!("   (In practice, these would come from actual detections)\n");

    // Simulated data: As threshold increases, precision increases but recall decreases
    let precisions = vec![0.50, 0.55, 0.62, 0.70, 0.78, 0.85, 0.90, 0.93, 0.95, 0.97, 1.00];
    let recalls = vec![1.00, 0.95, 0.92, 0.88, 0.82, 0.75, 0.68, 0.55, 0.40, 0.25, 0.10];

    println!("   Threshold | Precision | Recall | F1 Score");
    println!("   ----------|-----------|--------|----------");

    let mut f1_scores = Vec::new();
    for i in 0..thresholds.len() {
        let precision = precisions[i];
        let recall = recalls[i];
        let f1 = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };
        f1_scores.push(f1);

        println!(
            "   {:>8.2} | {:>9.4} | {:>6.4} | {:>8.4}",
            thresholds[i], precision, recall, f1
        );
    }
    println!();

    // Example 3: Find optimal threshold
    println!("3. Finding Optimal Threshold");
    let (best_f1, best_threshold) = find_optimal_f1_threshold(&precisions, &recalls, &thresholds);
    println!("   Best F1 Score: {:.4}", best_f1);
    println!("   Optimal Threshold: {:.2}", best_threshold);
    println!();

    // Example 4: Compute F1 scores using the library functions
    println!("4. Computing F1 Scores with Different Metrics");
    use coco_eval::metrics::f1_score::{calculate_f1_score, calculate_f1_from_counts};

    // Direct calculation
    let f1_direct = calculate_f1_score(0.85, 0.75);
    println!("   F1 from precision=0.85, recall=0.75: {:.4}", f1_direct);

    // From counts
    let f1_counts = calculate_f1_from_counts(17, 3, 6);
    println!("   F1 from TP=17, FP=3, FN=6:");
    println!("     Precision = 17/(17+3) = {:.4}", 17.0 / 20.0);
    println!("     Recall = 17/(17+6) = {:.4}", 17.0 / 23.0);
    println!("     F1 = {:.4}", f1_counts);
    println!();

    // Example 5: Threshold recommendations
    println!("5. Threshold Recommendations");
    println!("   Based on your use case:");
    println!();
    println!("   High Precision (low false positives):");
    println!("     - Recommended threshold: >= 0.7");
    println!("     - Use case: Medical diagnosis, security systems");
    println!();
    println!("   High Recall (low false negatives):");
    println!("     - Recommended threshold: <= 0.3");
    println!("     - Use case: Object detection for autonomous vehicles");
    println!();
    println!("   Balanced (optimal F1):");
    println!("     - Recommended threshold: {:.2}", best_threshold);
    println!("     - Use case: General object detection tasks");
    println!();

    // Example 6: Custom threshold ranges
    println!("6. Custom Threshold Ranges");

    let fine_range = generate_threshold_range(0.5, 0.9, 21)?;
    println!("   Fine-grained range (0.5 to 0.9, 21 steps):");
    println!("   First 5: {:?}", &fine_range[..5]);
    println!("   Last 5: {:?}", &fine_range[16..]);
    println!();

    let coarse_range = generate_threshold_range(0.0, 1.0, 5)?;
    println!("   Coarse range (0.0 to 1.0, 5 steps):");
    println!("   {:?}", coarse_range);
    println!();

    println!("=== Example Complete ===");

    Ok(())
}
