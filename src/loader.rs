//! JSON loading utilities for COCO format datasets.

use crate::error::{CocoEvalError, Result};
use crate::types::CocoDataset;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Load a COCO dataset from a JSON file.
///
/// # Arguments
///
/// * `path` - Path to the COCO JSON file
///
/// # Returns
///
/// Returns a `CocoDataset` containing images, annotations, and categories.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
///
/// # Example
///
/// ```no_run
/// use coco_eval::loader::load_from_file;
///
/// let dataset = load_from_file("annotations.json").unwrap();
/// println!("Loaded {} annotations", dataset.annotations.len());
/// ```
pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<CocoDataset> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let dataset: CocoDataset = serde_json::from_reader(reader)?;

    validate_dataset(&dataset)?;

    Ok(dataset)
}

/// Load a COCO dataset from a JSON string.
///
/// # Arguments
///
/// * `json_str` - JSON string in COCO format
///
/// # Returns
///
/// Returns a `CocoDataset` containing images, annotations, and categories.
///
/// # Errors
///
/// Returns an error if the JSON cannot be parsed.
///
/// # Example
///
/// ```
/// use coco_eval::loader::load_from_string;
///
/// let json = r#"{
///     "annotations": [],
///     "categories": [{"id": 1, "name": "person"}]
/// }"#;
/// let dataset = load_from_string(json).unwrap();
/// ```
pub fn load_from_string(json_str: &str) -> Result<CocoDataset> {
    let dataset: CocoDataset = serde_json::from_str(json_str)?;
    validate_dataset(&dataset)?;
    Ok(dataset)
}

/// Validate that a COCO dataset has the required structure.
fn validate_dataset(dataset: &CocoDataset) -> Result<()> {
    if dataset.categories.is_empty() {
        return Err(CocoEvalError::EmptyDataset(
            "Dataset must contain at least one category".to_string()
        ));
    }

    // Validate that all annotations have valid bbox format
    for annotation in &dataset.annotations {
        if annotation.bbox.len() != 4 {
            return Err(CocoEvalError::InvalidAnnotation(
                format!("Annotation {} has invalid bbox length: {}",
                    annotation.id, annotation.bbox.len())
            ));
        }

        // Check for non-negative dimensions
        if annotation.bbox[2] < 0.0 || annotation.bbox[3] < 0.0 {
            return Err(CocoEvalError::InvalidBoundingBox(
                format!("Annotation {} has negative dimensions", annotation.id)
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_from_string() {
        let json = r#"{
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10.0, 20.0, 30.0, 40.0]
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person"
                }
            ]
        }"#;

        let dataset = load_from_string(json).unwrap();
        assert_eq!(dataset.annotations.len(), 1);
        assert_eq!(dataset.categories.len(), 1);
    }

    #[test]
    fn test_empty_categories() {
        let json = r#"{
            "annotations": [],
            "categories": []
        }"#;

        let result = load_from_string(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_bbox() {
        let json = r#"{
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10.0, 20.0, 30.0]
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "person"
                }
            ]
        }"#;

        let result = load_from_string(json);
        assert!(result.is_err());
    }
}
