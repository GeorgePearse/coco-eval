//! Confidence score thresholding utilities.

use crate::error::{CocoEvalError, Result};
use crate::types::{Annotation, CocoDataset};

/// Filter annotations by confidence score threshold.
///
/// # Arguments
///
/// * `annotations` - Vector of annotations to filter
/// * `threshold` - Minimum confidence score (0.0 to 1.0)
///
/// # Returns
///
/// Returns a new vector containing only annotations with score >= threshold.
///
/// # Errors
///
/// Returns an error if the threshold is not in the valid range [0.0, 1.0].
///
/// # Example
///
/// ```
/// use coco_eval::threshold::filter_by_confidence;
/// use coco_eval::types::Annotation;
///
/// let annotations = vec![
///     Annotation {
///         id: 1,
///         image_id: 1,
///         category_id: 1,
///         bbox: vec![10.0, 20.0, 30.0, 40.0],
///         area: None,
///         iscrowd: None,
///         score: Some(0.9),
///     },
///     Annotation {
///         id: 2,
///         image_id: 1,
///         category_id: 1,
///         bbox: vec![50.0, 60.0, 70.0, 80.0],
///         area: None,
///         iscrowd: None,
///         score: Some(0.3),
///     },
/// ];
///
/// let filtered = filter_by_confidence(&annotations, 0.5).unwrap();
/// assert_eq!(filtered.len(), 1);
/// ```
pub fn filter_by_confidence(annotations: &[Annotation], threshold: f64) -> Result<Vec<Annotation>> {
    validate_threshold(threshold)?;

    Ok(annotations
        .iter()
        .filter(|ann| ann.confidence() >= threshold)
        .cloned()
        .collect())
}

/// Filter a COCO dataset by confidence threshold.
///
/// Creates a new dataset with only annotations passing the threshold.
///
/// # Arguments
///
/// * `dataset` - The COCO dataset to filter
/// * `threshold` - Minimum confidence score (0.0 to 1.0)
///
/// # Returns
///
/// Returns a new `CocoDataset` with filtered annotations.
///
/// # Errors
///
/// Returns an error if the threshold is not in the valid range [0.0, 1.0].
pub fn filter_dataset_by_confidence(dataset: &CocoDataset, threshold: f64) -> Result<CocoDataset> {
    validate_threshold(threshold)?;

    let filtered_annotations = filter_by_confidence(&dataset.annotations, threshold)?;

    Ok(CocoDataset {
        images: dataset.images.clone(),
        annotations: filtered_annotations,
        categories: dataset.categories.clone(),
    })
}

/// Generate a range of threshold values for evaluation.
///
/// # Arguments
///
/// * `start` - Starting threshold value (inclusive)
/// * `end` - Ending threshold value (inclusive)
/// * `steps` - Number of threshold values to generate
///
/// # Returns
///
/// Returns a vector of evenly-spaced threshold values.
///
/// # Example
///
/// ```
/// use coco_eval::threshold::generate_threshold_range;
///
/// let thresholds = generate_threshold_range(0.0, 1.0, 11).unwrap();
/// assert_eq!(thresholds.len(), 11);
/// assert_eq!(thresholds[0], 0.0);
/// assert_eq!(thresholds[10], 1.0);
/// ```
pub fn generate_threshold_range(start: f64, end: f64, steps: usize) -> Result<Vec<f64>> {
    if steps == 0 {
        return Err(CocoEvalError::InvalidThreshold(
            "Number of steps must be greater than 0".to_string()
        ));
    }

    validate_threshold(start)?;
    validate_threshold(end)?;

    if start > end {
        return Err(CocoEvalError::InvalidThreshold(
            format!("Start threshold ({}) must be <= end threshold ({})", start, end)
        ));
    }

    if steps == 1 {
        return Ok(vec![start]);
    }

    let step_size = (end - start) / (steps - 1) as f64;
    Ok((0..steps)
        .map(|i| start + step_size * i as f64)
        .collect())
}

/// Validate that a threshold is in the valid range [0.0, 1.0].
fn validate_threshold(threshold: f64) -> Result<()> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(CocoEvalError::InvalidThreshold(
            format!("Threshold must be between 0.0 and 1.0, got {}", threshold)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_by_confidence() {
        let annotations = vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: vec![10.0, 20.0, 30.0, 40.0],
                area: None,
                iscrowd: None,
                score: Some(0.9),
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 1,
                bbox: vec![50.0, 60.0, 70.0, 80.0],
                area: None,
                iscrowd: None,
                score: Some(0.3),
            },
        ];

        let filtered = filter_by_confidence(&annotations, 0.5).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].id, 1);
    }

    #[test]
    fn test_invalid_threshold() {
        let annotations = vec![];
        assert!(filter_by_confidence(&annotations, 1.5).is_err());
        assert!(filter_by_confidence(&annotations, -0.1).is_err());
    }

    #[test]
    fn test_generate_threshold_range() {
        let thresholds = generate_threshold_range(0.0, 1.0, 11).unwrap();
        assert_eq!(thresholds.len(), 11);
        assert!((thresholds[0] - 0.0).abs() < 1e-10);
        assert!((thresholds[10] - 1.0).abs() < 1e-10);
        assert!((thresholds[5] - 0.5).abs() < 1e-10);
    }
}
