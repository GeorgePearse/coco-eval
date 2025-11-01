//! Core data types for COCO annotations and evaluations.

use serde::{Deserialize, Serialize};

/// Represents a bounding box in COCO format (x, y, width, height).
///
/// Coordinates are in LTWH (Left-Top-Width-Height) format where:
/// - x: Left coordinate
/// - y: Top coordinate
/// - width: Box width
/// - height: Box height
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl BoundingBox {
    /// Create a new bounding box.
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self { x, y, width, height }
    }

    /// Get the area of the bounding box.
    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    /// Get the right coordinate (x + width).
    pub fn right(&self) -> f64 {
        self.x + self.width
    }

    /// Get the bottom coordinate (y + height).
    pub fn bottom(&self) -> f64 {
        self.y + self.height
    }

    /// Check if the bounding box is valid (positive dimensions).
    pub fn is_valid(&self) -> bool {
        self.width > 0.0 && self.height > 0.0
    }
}

/// Represents a category in the COCO dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Category {
    pub id: u64,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supercategory: Option<String>,
}

/// Represents an image in the COCO dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Image {
    pub id: u64,
    pub file_name: String,
    pub height: u32,
    pub width: u32,
}

/// Represents an annotation in COCO format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    pub id: u64,
    pub image_id: u64,
    pub category_id: u64,
    /// Bounding box in [x, y, width, height] format
    pub bbox: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub area: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iscrowd: Option<u8>,
    /// Confidence score (for predictions)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

impl Annotation {
    /// Convert the bbox array to a BoundingBox struct.
    pub fn to_bbox(&self) -> crate::error::Result<BoundingBox> {
        if self.bbox.len() != 4 {
            return Err(crate::error::CocoEvalError::InvalidBoundingBox(
                format!("Expected 4 values, got {}", self.bbox.len())
            ));
        }
        Ok(BoundingBox::new(
            self.bbox[0],
            self.bbox[1],
            self.bbox[2],
            self.bbox[3],
        ))
    }

    /// Get the confidence score, defaulting to 1.0 if not present.
    pub fn confidence(&self) -> f64 {
        self.score.unwrap_or(1.0)
    }
}

/// Represents a complete COCO dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CocoDataset {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<Image>>,
    pub annotations: Vec<Annotation>,
    pub categories: Vec<Category>,
}

/// Detection result for a single image.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub image_id: u64,
    pub category_id: u64,
    pub bbox: BoundingBox,
    pub score: f64,
    pub is_true_positive: bool,
    pub matched_gt_id: Option<u64>,
}

/// Evaluation metrics for object detection.
#[derive(Debug, Clone, Default)]
pub struct EvaluationMetrics {
    /// Mean Average Precision across all classes
    pub map: f64,
    /// Average Precision at IoU=0.50
    pub ap50: f64,
    /// Average Precision at IoU=0.75
    pub ap75: f64,
    /// Per-class Average Precision values
    pub ap_per_class: Vec<(u64, f64)>,
    /// Precision at various confidence thresholds
    pub precision_at_thresholds: Vec<(f64, f64)>,
    /// Recall at various confidence thresholds
    pub recall_at_thresholds: Vec<(f64, f64)>,
    /// F1 score at various confidence thresholds
    pub f1_at_thresholds: Vec<(f64, f64)>,
}

impl EvaluationMetrics {
    /// Create a new empty EvaluationMetrics.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Precision-Recall curve point.
#[derive(Debug, Clone)]
pub struct PrecisionRecallPoint {
    pub precision: f64,
    pub recall: f64,
    pub threshold: f64,
}
