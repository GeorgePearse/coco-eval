/// Statistics tracking for inference/prediction processing
///
/// This module provides structures and utilities for tracking statistics during
/// the validation and processing of predictions.

use serde::{Deserialize, Serialize};

/// Statistics collected during prediction processing
///
/// Tracks validation failures, successful processing, and other metrics
/// encountered during the preparation of predictions for evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceStats {
    /// Total number of predictions processed
    pub total_predictions: usize,

    /// Number of predictions skipped due to invalid bounding boxes
    pub skipped_invalid_boxes: usize,

    /// Number of predictions skipped due to invalid/unknown categories
    pub skipped_invalid_categories: usize,

    /// Number of predictions skipped due to missing image IDs
    pub skipped_missing_images: usize,

    /// Number of predictions skipped due to unavailable class
    pub skipped_unavailable_class: usize,

    /// Number of unique images with predictions processed
    pub processed_images: usize,

    /// Number of images with zero predictions after filtering
    pub empty_predictions: usize,
}

impl InferenceStats {
    /// Create a new `InferenceStats` with all counters at zero
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment the total predictions counter
    pub fn add_prediction(&mut self) {
        self.total_predictions += 1;
    }

    /// Record a prediction skipped due to invalid bounding box
    pub fn skip_invalid_box(&mut self) {
        self.skipped_invalid_boxes += 1;
    }

    /// Record a prediction skipped due to invalid category
    pub fn skip_invalid_category(&mut self) {
        self.skipped_invalid_categories += 1;
    }

    /// Record a prediction skipped due to missing image
    pub fn skip_missing_image(&mut self) {
        self.skipped_missing_images += 1;
    }

    /// Record a prediction skipped due to unavailable class
    pub fn skip_unavailable_class(&mut self) {
        self.skipped_unavailable_class += 1;
    }

    /// Set the number of processed images
    pub fn set_processed_images(&mut self, count: usize) {
        self.processed_images = count;
    }

    /// Set the number of empty prediction images
    pub fn set_empty_predictions(&mut self, count: usize) {
        self.empty_predictions = count;
    }

    /// Calculate the number of valid predictions
    ///
    /// Returns the number of predictions that passed all validation checks
    pub fn valid_predictions(&self) -> usize {
        self.total_predictions
            .saturating_sub(self.skipped_invalid_boxes)
            .saturating_sub(self.skipped_invalid_categories)
            .saturating_sub(self.skipped_missing_images)
            .saturating_sub(self.skipped_unavailable_class)
    }

    /// Calculate the total number of skipped predictions
    pub fn total_skipped(&self) -> usize {
        self.skipped_invalid_boxes
            + self.skipped_invalid_categories
            + self.skipped_missing_images
            + self.skipped_unavailable_class
    }

    /// Print a summary of the statistics to stdout
    ///
    /// Displays a formatted summary of all tracked statistics
    pub fn print_summary(&self) {
        println!("\n=== Inference Statistics ===");
        println!("Total predictions: {}", self.total_predictions);
        println!("Valid predictions: {}", self.valid_predictions());
        println!("Total skipped: {}", self.total_skipped());
        println!("  - Invalid boxes: {}", self.skipped_invalid_boxes);
        println!("  - Invalid categories: {}", self.skipped_invalid_categories);
        println!("  - Missing images: {}", self.skipped_missing_images);
        println!("  - Unavailable class: {}", self.skipped_unavailable_class);
        println!("Processed images: {}", self.processed_images);
        println!("Empty prediction images: {}", self.empty_predictions);
        println!("===========================\n");
    }

    /// Get a formatted string summary of the statistics
    pub fn summary_string(&self) -> String {
        format!(
            "InferenceStats {{ total: {}, valid: {}, skipped: {}, processed_images: {}, empty: {} }}",
            self.total_predictions,
            self.valid_predictions(),
            self.total_skipped(),
            self.processed_images,
            self.empty_predictions
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_stats_are_zero() {
        let stats = InferenceStats::new();
        assert_eq!(stats.total_predictions, 0);
        assert_eq!(stats.valid_predictions(), 0);
        assert_eq!(stats.total_skipped(), 0);
    }

    #[test]
    fn test_add_prediction() {
        let mut stats = InferenceStats::new();
        stats.add_prediction();
        stats.add_prediction();
        assert_eq!(stats.total_predictions, 2);
    }

    #[test]
    fn test_skip_counters() {
        let mut stats = InferenceStats::new();
        stats.skip_invalid_box();
        stats.skip_invalid_category();
        stats.skip_missing_image();
        stats.skip_unavailable_class();

        assert_eq!(stats.skipped_invalid_boxes, 1);
        assert_eq!(stats.skipped_invalid_categories, 1);
        assert_eq!(stats.skipped_missing_images, 1);
        assert_eq!(stats.skipped_unavailable_class, 1);
        assert_eq!(stats.total_skipped(), 4);
    }

    #[test]
    fn test_valid_predictions() {
        let mut stats = InferenceStats::new();
        stats.total_predictions = 100;
        stats.skipped_invalid_boxes = 10;
        stats.skipped_invalid_categories = 5;

        assert_eq!(stats.valid_predictions(), 85);
    }

    #[test]
    fn test_summary_string() {
        let mut stats = InferenceStats::new();
        stats.total_predictions = 50;
        stats.processed_images = 10;

        let summary = stats.summary_string();
        assert!(summary.contains("total: 50"));
        assert!(summary.contains("processed_images: 10"));
    }
}
