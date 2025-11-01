/// Utilities for working with Polars DataFrames
///
/// This module provides helper functions for validating, manipulating, and
/// converting Polars DataFrames, particularly for COCO evaluation tasks.

use polars::prelude::*;
use crate::error::CocoEvalError;

/// Validate that a DataFrame contains all required columns
///
/// # Arguments
///
/// * `df` - The DataFrame to validate
/// * `required_columns` - Slice of required column names
///
/// # Returns
///
/// `Ok(())` if all columns are present, error otherwise
pub fn validate_columns(df: &DataFrame, required_columns: &[&str]) -> Result<(), CocoEvalError> {
    let column_names: Vec<String> = df.get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    for col in required_columns {
        if !column_names.iter().any(|c| c == col) {
            return Err(CocoEvalError::MissingColumn(col.to_string()));
        }
    }

    Ok(())
}

/// Validate the schema of an annotations DataFrame
///
/// Expected columns: image_id, category_id, bbox
pub fn validate_annotations_schema(df: &DataFrame) -> Result<(), CocoEvalError> {
    validate_columns(df, &["image_id", "category_id", "bbox"])?;

    // Check types
    let image_id_dtype = df.column("image_id")?.dtype();
    if !matches!(image_id_dtype, DataType::Int64 | DataType::Int32 | DataType::UInt64 | DataType::UInt32) {
        return Err(CocoEvalError::InvalidDataFrame(
            format!("image_id must be integer type, got {:?}", image_id_dtype)
        ));
    }

    let category_id_dtype = df.column("category_id")?.dtype();
    if !matches!(category_id_dtype, DataType::Int64 | DataType::Int32) {
        return Err(CocoEvalError::InvalidDataFrame(
            format!("category_id must be Int64 or Int32, got {:?}", category_id_dtype)
        ));
    }

    Ok(())
}

/// Validate the schema of a predictions DataFrame
///
/// Expected columns: image_id, category_id, bbox, score
pub fn validate_predictions_schema(df: &DataFrame) -> Result<(), CocoEvalError> {
    validate_columns(df, &["image_id", "category_id", "bbox", "score"])?;

    // Check score column type
    let score_dtype = df.column("score")?.dtype();
    if !matches!(score_dtype, DataType::Float64 | DataType::Float32) {
        return Err(CocoEvalError::InvalidDataFrame(
            format!("score must be Float64 or Float32, got {:?}", score_dtype)
        ));
    }

    Ok(())
}

/// Validate the schema of an images DataFrame
///
/// Expected columns: image_id, file_name
pub fn validate_images_schema(df: &DataFrame) -> Result<(), CocoEvalError> {
    validate_columns(df, &["image_id", "file_name"])?;

    let file_name_dtype = df.column("file_name")?.dtype();
    if !matches!(file_name_dtype, DataType::String) {
        return Err(CocoEvalError::InvalidDataFrame(
            format!("file_name must be String, got {:?}", file_name_dtype)
        ));
    }

    Ok(())
}

/// Extract a bounding box list from a Polars Series at a given index
///
/// # Arguments
///
/// * `bbox_series` - Series containing bbox data (expected to be List type)
/// * `idx` - Index of the bbox to extract
///
/// # Returns
///
/// A Vec of 4 f64 values representing [x, y, width, height]
pub fn extract_bbox_from_series(bbox_series: &Series, idx: usize) -> Result<Vec<f64>, CocoEvalError> {
    // Handle List type
    if let Ok(list_ca) = bbox_series.list() {
        let bbox_series = list_ca.get_as_series(idx)
            .ok_or_else(|| CocoEvalError::InvalidDataFrame(
                format!("Could not extract bbox at index {}", idx)
            ))?;

        let bbox_values = bbox_series.f64()?;
        let bbox: Vec<f64> = bbox_values.into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();

        if bbox.len() != 4 {
            return Err(CocoEvalError::InvalidBoundingBox(
                format!("Bbox must have 4 elements, got {}", bbox.len())
            ));
        }

        Ok(bbox)
    } else {
        Err(CocoEvalError::InvalidDataFrame(
            "bbox column must be of List type".to_string()
        ))
    }
}

/// Convert NaN values to zeros in a DataFrame column
///
/// This is useful when Python/Pandas DataFrames with NaN values are passed
/// and need to be cleaned for computation.
pub fn nan_to_zeros(series: &Series) -> Result<Series, CocoEvalError> {
    match series.dtype() {
        DataType::Float64 => {
            let ca = series.f64()?;
            let filled = ca.fill_null_with_values(0.0)?;
            Ok(filled.into_series())
        },
        DataType::Float32 => {
            let ca = series.f32()?;
            let filled = ca.fill_null_with_values(0.0)?;
            Ok(filled.into_series())
        },
        _ => Ok(series.clone())
    }
}

/// Filter DataFrame rows where a column value is in a given set
///
/// # Arguments
///
/// * `df` - The DataFrame to filter
/// * `_column` - Name of the column to filter on
/// * `_valid_values` - Set of valid values to keep
///
/// # Returns
///
/// A new DataFrame with only rows where `column` value is in `valid_values`
///
/// # Note
///
/// This is currently a stub implementation that returns the original DataFrame.
/// Full implementation will be added when needed by the evaluator.
pub fn filter_by_set<T>(
    df: &DataFrame,
    _column: &str,
    _valid_values: &std::collections::HashSet<T>,
) -> Result<DataFrame, CocoEvalError>
where
    T: std::cmp::Eq + std::hash::Hash + Clone + std::fmt::Display,
{
    // This is a simplified version - in practice, we'd need type-specific handling
    // For now, we'll return the original DataFrame
    // TODO: Implement proper filtering based on type
    Ok(df.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_columns_success() {
        let df = df! {
            "col1" => &[1, 2, 3],
            "col2" => &["a", "b", "c"],
        }.unwrap();

        assert!(validate_columns(&df, &["col1", "col2"]).is_ok());
    }

    #[test]
    fn test_validate_columns_missing() {
        let df = df! {
            "col1" => &[1, 2, 3],
        }.unwrap();

        let result = validate_columns(&df, &["col1", "col2"]);
        assert!(result.is_err());
        match result {
            Err(CocoEvalError::MissingColumn(col)) => assert_eq!(col, "col2"),
            _ => panic!("Expected MissingColumn error"),
        }
    }

    #[test]
    fn test_nan_to_zeros() {
        let series = Series::new("test".into(), &[Some(1.0), None, Some(3.0)]);
        let result = nan_to_zeros(&series).unwrap();

        let values: Vec<f64> = result.f64().unwrap().into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();

        assert_eq!(values, vec![1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_extract_bbox_from_series() {
        // Create a Series with a list of bboxes
        let bbox1 = Series::new("".into(), &[10.0, 20.0, 30.0, 40.0]);
        let bbox2 = Series::new("".into(), &[50.0, 60.0, 70.0, 80.0]);

        let list_series = Series::new("bbox".into(), &[bbox1, bbox2]);

        let extracted = extract_bbox_from_series(&list_series, 0).unwrap();
        assert_eq!(extracted, vec![10.0, 20.0, 30.0, 40.0]);

        let extracted2 = extract_bbox_from_series(&list_series, 1).unwrap();
        assert_eq!(extracted2, vec![50.0, 60.0, 70.0, 80.0]);
    }
}
