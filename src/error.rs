//! Error types for the coco-eval library.

use thiserror::Error;

/// Result type for coco-eval operations.
pub type Result<T> = std::result::Result<T, CocoEvalError>;

/// Error types that can occur during COCO evaluation.
#[derive(Error, Debug)]
pub enum CocoEvalError {
    /// Error during JSON parsing or serialization.
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// Error during I/O operations.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Invalid annotation data.
    #[error("Invalid annotation: {0}")]
    InvalidAnnotation(String),

    /// Invalid bounding box coordinates.
    #[error("Invalid bounding box: {0}")]
    InvalidBoundingBox(String),

    /// Mismatched categories between ground truth and predictions.
    #[error("Category mismatch: {0}")]
    CategoryMismatch(String),

    /// Missing required field in COCO format.
    #[error("Missing field: {0}")]
    MissingField(String),

    /// Empty dataset provided.
    #[error("Empty dataset: {0}")]
    EmptyDataset(String),

    /// Invalid confidence threshold.
    #[error("Invalid threshold: {0}")]
    InvalidThreshold(String),
}
