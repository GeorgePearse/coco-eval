/// Non-Maximum Suppression (`NMS`) implementation
///
/// This module provides `NMS` functionality for filtering overlapping bounding boxes,
/// matching the behavior of the Python supervision library.

use crate::error::CocoEvalError;

/// Bounding box in [x, y, width, height] format
pub type BBoxXYWH = [f64; 4];

/// Bounding box in [x1, y1, x2, y2] format
pub type BBoxXYXY = [f64; 4];

/// Convert bounding box from [x, y, width, height] to [x1, y1, x2, y2]
///
/// # Arguments
///
/// * `bbox` - Bounding box in XYWH format
///
/// # Returns
///
/// Bounding box in XYXY format
///
/// # Examples
///
/// ```
/// # use coco_eval::nms::xywh_to_xyxy;
/// let xywh = [10.0, 20.0, 30.0, 40.0];
/// let xyxy = xywh_to_xyxy(xywh);
/// assert_eq!(xyxy, [10.0, 20.0, 40.0, 60.0]);
/// ```
#[must_use]
pub fn xywh_to_xyxy(bbox: BBoxXYWH) -> BBoxXYXY {
    [
        bbox[0],              // x1 = x
        bbox[1],              // y1 = y
        bbox[0] + bbox[2],    // x2 = x + width
        bbox[1] + bbox[3],    // y2 = y + height
    ]
}

/// Convert bounding box from [x1, y1, x2, y2] to [x, y, width, height]
///
/// # Arguments
///
/// * `bbox` - Bounding box in XYXY format
///
/// # Returns
///
/// Bounding box in XYWH format
///
/// # Examples
///
/// ```
/// # use coco_eval::nms::xyxy_to_xywh;
/// let xyxy = [10.0, 20.0, 40.0, 60.0];
/// let xywh = xyxy_to_xywh(xyxy);
/// assert_eq!(xywh, [10.0, 20.0, 30.0, 40.0]);
/// ```
#[must_use]
pub fn xyxy_to_xywh(bbox: BBoxXYXY) -> BBoxXYWH {
    [
        bbox[0],              // x = x1
        bbox[1],              // y = y1
        bbox[2] - bbox[0],    // width = x2 - x1
        bbox[3] - bbox[1],    // height = y2 - y1
    ]
}

/// Calculate Intersection over Union (`IoU`) between two boxes in `XYXY` format
///
/// # Arguments
///
/// * `box1` - First bounding box in `XYXY` format
/// * `box2` - Second bounding box in `XYXY` format
///
/// # Returns
///
/// `IoU` value between 0.0 and 1.0
#[must_use]
fn calculate_iou_xyxy(box1: &BBoxXYXY, box2: &BBoxXYXY) -> f64 {
    // Calculate intersection coordinates
    let x1 = box1[0].max(box2[0]);
    let y1 = box1[1].max(box2[1]);
    let x2 = box1[2].min(box2[2]);
    let y2 = box1[3].min(box2[3]);

    // Calculate intersection area
    let intersection_width = (x2 - x1).max(0.0);
    let intersection_height = (y2 - y1).max(0.0);
    let intersection_area = intersection_width * intersection_height;

    // Calculate union area
    let box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    let union_area = box1_area + box2_area - intersection_area;

    if union_area == 0.0 {
        return 0.0;
    }

    intersection_area / union_area
}

/// Detection with bounding box and confidence score
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box in XYXY format
    pub bbox: BBoxXYXY,
    /// Confidence score
    pub score: f64,
    /// Original index in the input
    pub index: usize,
}

/// Apply Non-Maximum Suppression to a set of detections
///
/// This implementation matches the behavior of `supervision.box_non_max_suppression`.
///
/// # Arguments
///
/// * `detections` - Vector of detections with bboxes and scores
/// * `iou_threshold` - `IoU` threshold for suppression (0.0 to 1.0)
///
/// # Returns
///
/// Boolean mask indicating which detections to keep
///
/// # Errors
///
/// Returns error if `iou_threshold` is not in range [0.0, 1.0]
///
/// # Examples
///
/// ```
/// # use coco_eval::nms::{Detection, non_maximum_suppression};
/// let detections = vec![
///     Detection { bbox: [10.0, 10.0, 50.0, 50.0], score: 0.9, index: 0 },
///     Detection { bbox: [15.0, 15.0, 55.0, 55.0], score: 0.8, index: 1 },
///     Detection { bbox: [100.0, 100.0, 150.0, 150.0], score: 0.95, index: 2 },
/// ];
///
/// let keep_mask = non_maximum_suppression(&detections, 0.5).unwrap();
/// assert_eq!(keep_mask, vec![true, false, true]);
/// ```
pub fn non_maximum_suppression(
    detections: &[Detection],
    iou_threshold: f64,
) -> Result<Vec<bool>, CocoEvalError> {
    if !(0.0..=1.0).contains(&iou_threshold) {
        return Err(CocoEvalError::InvalidThreshold(
            format!("IoU threshold must be between 0 and 1, got {iou_threshold}")
        ));
    }

    let n = detections.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Initialize all as kept
    let mut keep_mask = vec![true; n];

    // Sort indices by score (descending)
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        detections[b].score
            .partial_cmp(&detections[a].score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Process each detection in order of decreasing score
    for (i, &idx_i) in indices.iter().enumerate() {
        if !keep_mask[idx_i] {
            continue;
        }

        // Suppress all subsequent boxes that have high IoU with this one
        for &idx_j in &indices[(i + 1)..] {
            if !keep_mask[idx_j] {
                continue;
            }

            let iou = calculate_iou_xyxy(&detections[idx_i].bbox, &detections[idx_j].bbox);

            if iou > iou_threshold {
                keep_mask[idx_j] = false;
            }
        }
    }

    Ok(keep_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xywh_to_xyxy() {
        let xywh = [10.0, 20.0, 30.0, 40.0];
        let xyxy = xywh_to_xyxy(xywh);
        assert_eq!(xyxy, [10.0, 20.0, 40.0, 60.0]);
    }

    #[test]
    fn test_xyxy_to_xywh() {
        let xyxy = [10.0, 20.0, 40.0, 60.0];
        let xywh = xyxy_to_xywh(xyxy);
        assert_eq!(xywh, [10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_bbox_conversion_roundtrip() {
        let original = [5.5, 10.3, 25.7, 30.2];
        let xyxy = xywh_to_xyxy(original);
        let xywh = xyxy_to_xywh(xyxy);

        for i in 0..4 {
            assert!((original[i] - xywh[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_calculate_iou_xyxy_identical() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [0.0, 0.0, 10.0, 10.0];
        let iou = calculate_iou_xyxy(&box1, &box2);
        assert!((iou - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_iou_xyxy_no_overlap() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [20.0, 20.0, 30.0, 30.0];
        let iou = calculate_iou_xyxy(&box1, &box2);
        assert_eq!(iou, 0.0);
    }

    #[test]
    fn test_calculate_iou_xyxy_partial_overlap() {
        let box1 = [0.0, 0.0, 10.0, 10.0];
        let box2 = [5.0, 5.0, 15.0, 15.0];
        let iou = calculate_iou_xyxy(&box1, &box2);

        // Intersection: 5x5 = 25
        // Union: 100 + 100 - 25 = 175
        // IoU: 25/175 = 0.142857...
        assert!((iou - 0.142857).abs() < 1e-5);
    }

    #[test]
    fn test_nms_empty_input() {
        let detections: Vec<Detection> = vec![];
        let keep_mask = non_maximum_suppression(&detections, 0.5).unwrap();
        assert_eq!(keep_mask.len(), 0);
    }

    #[test]
    fn test_nms_single_detection() {
        let detections = vec![
            Detection {
                bbox: [0.0, 0.0, 10.0, 10.0],
                score: 0.9,
                index: 0,
            }
        ];
        let keep_mask = non_maximum_suppression(&detections, 0.5).unwrap();
        assert_eq!(keep_mask, vec![true]);
    }

    #[test]
    fn test_nms_no_overlap() {
        let detections = vec![
            Detection {
                bbox: [0.0, 0.0, 10.0, 10.0],
                score: 0.9,
                index: 0,
            },
            Detection {
                bbox: [20.0, 20.0, 30.0, 30.0],
                score: 0.8,
                index: 1,
            },
        ];
        let keep_mask = non_maximum_suppression(&detections, 0.5).unwrap();
        assert_eq!(keep_mask, vec![true, true]);
    }

    #[test]
    fn test_nms_high_overlap() {
        // Two highly overlapping boxes, should keep only the one with higher score
        let detections = vec![
            Detection {
                bbox: [10.0, 10.0, 50.0, 50.0],
                score: 0.9,
                index: 0,
            },
            Detection {
                bbox: [15.0, 15.0, 55.0, 55.0],
                score: 0.8,
                index: 1,
            },
        ];
        let keep_mask = non_maximum_suppression(&detections, 0.5).unwrap();
        assert_eq!(keep_mask, vec![true, false]);
    }

    #[test]
    fn test_nms_complex_scenario() {
        let detections = vec![
            Detection {
                bbox: [10.0, 10.0, 50.0, 50.0],
                score: 0.9,
                index: 0,
            },
            Detection {
                bbox: [15.0, 15.0, 55.0, 55.0],
                score: 0.8,
                index: 1,
            },
            Detection {
                bbox: [100.0, 100.0, 150.0, 150.0],
                score: 0.95,
                index: 2,
            },
            Detection {
                bbox: [105.0, 105.0, 155.0, 155.0],
                score: 0.7,
                index: 3,
            },
        ];
        let keep_mask = non_maximum_suppression(&detections, 0.5).unwrap();

        // Should keep 0 (high score), suppress 1 (overlaps with 0)
        // Should keep 2 (highest score, different area), suppress 3 (overlaps with 2 with IoU > 0.5)
        // IoU between 2 and 3: intersection 45x45=2025, union 2500+2500-2025=2975, IoU=0.68 > 0.5
        assert_eq!(keep_mask, vec![true, false, true, false]);
    }

    #[test]
    fn test_nms_invalid_threshold() {
        let detections = vec![
            Detection {
                bbox: [0.0, 0.0, 10.0, 10.0],
                score: 0.9,
                index: 0,
            }
        ];

        // Test threshold > 1.0
        let result = non_maximum_suppression(&detections, 1.5);
        assert!(result.is_err());

        // Test threshold < 0.0
        let result = non_maximum_suppression(&detections, -0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_nms_score_ordering() {
        // Even if lower-score box comes first, higher-score should be kept
        let detections = vec![
            Detection {
                bbox: [10.0, 10.0, 50.0, 50.0],
                score: 0.7,
                index: 0,
            },
            Detection {
                bbox: [15.0, 15.0, 55.0, 55.0],
                score: 0.9,
                index: 1,
            },
        ];
        let keep_mask = non_maximum_suppression(&detections, 0.5).unwrap();

        // Higher score (0.9) should be kept, lower score (0.7) should be suppressed
        assert_eq!(keep_mask, vec![false, true]);
    }
}
