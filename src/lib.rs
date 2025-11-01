//! # coco-eval
//!
//! A Rust library for COCO (Common Objects in Context) object detection evaluation metrics
//! with confidence thresholding support.
//!
//! This library provides implementations of standard object detection metrics including:
//! - **mAP** (mean Average Precision)
//! - **AP50** (Average Precision at IoU=0.50)
//! - **AP75** (Average Precision at IoU=0.75)
//! - **Precision** at various confidence thresholds
//! - **Recall** at various confidence thresholds
//! - **F1 Score** at various confidence thresholds
//!
//! ## Features
//!
//! - Load and parse COCO format JSON annotations
//! - Calculate IoU (Intersection over Union) between bounding boxes
//! - Compute precision, recall, and F1 scores
//! - Calculate Average Precision (AP) and mean Average Precision (mAP)
//! - Filter predictions by confidence score thresholds
//! - Evaluate at multiple confidence and IoU thresholds
//!
//! ## Quick Start
//!
//! ```rust
//! use coco_eval::loader::load_from_file;
//! use coco_eval::evaluator::evaluate;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load ground truth and predictions (in actual use)
//! // let ground_truth = load_from_file("ground_truth.json")?;
//! // let predictions = load_from_file("predictions.json")?;
//!
//! // Evaluate with default thresholds
//! // let metrics = evaluate(&ground_truth, &predictions, None, None)?;
//!
//! // Print results
//! // println!("mAP: {:.4}", metrics.map);
//! // println!("AP50: {:.4}", metrics.ap50);
//! // println!("AP75: {:.4}", metrics.ap75);
//! # Ok(())
//! # }
//! ```
//!
//! ## COCO Format
//!
//! This library expects annotations in the standard COCO JSON format:
//!
//! ```json
//! {
//!   "images": [...],
//!   "annotations": [
//!     {
//!       "id": 1,
//!       "image_id": 1,
//!       "category_id": 1,
//!       "bbox": [x, y, width, height],
//!       "score": 0.95  // For predictions only
//!     }
//!   ],
//!   "categories": [
//!     {
//!       "id": 1,
//!       "name": "person"
//!     }
//!   ]
//! }
//! ```

pub mod error;
pub mod types;
pub mod loader;
pub mod threshold;
pub mod metrics;
pub mod matching;
pub mod evaluator;

// Re-export commonly used types and functions
pub use error::{CocoEvalError, Result};
pub use types::{
    BoundingBox, Annotation, Category, Image, CocoDataset,
    EvaluationMetrics, DetectionResult, PrecisionRecallPoint,
};
pub use loader::{load_from_file, load_from_string};
pub use threshold::{filter_by_confidence, filter_dataset_by_confidence, generate_threshold_range};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_compiles() {
        // Basic smoke test to ensure the library compiles
        let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(bbox.is_valid());
    }
}
