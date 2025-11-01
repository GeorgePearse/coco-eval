//! Intersection over Union (IoU) calculation.

use crate::types::BoundingBox;

/// Calculate the Intersection over Union (IoU) between two bounding boxes.
///
/// IoU is defined as the area of intersection divided by the area of union.
///
/// # Arguments
///
/// * `bbox1` - First bounding box
/// * `bbox2` - Second bounding box
///
/// # Returns
///
/// Returns a value between 0.0 (no overlap) and 1.0 (perfect overlap).
///
/// # Example
///
/// ```
/// use coco_eval::metrics::iou::calculate_iou;
/// use coco_eval::types::BoundingBox;
///
/// let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
/// let bbox2 = BoundingBox::new(5.0, 5.0, 10.0, 10.0);
/// let iou = calculate_iou(&bbox1, &bbox2);
/// assert!(iou > 0.0 && iou < 1.0);
/// ```
pub fn calculate_iou(bbox1: &BoundingBox, bbox2: &BoundingBox) -> f64 {
    // Calculate intersection coordinates
    let x_left = bbox1.x.max(bbox2.x);
    let y_top = bbox1.y.max(bbox2.y);
    let x_right = bbox1.right().min(bbox2.right());
    let y_bottom = bbox1.bottom().min(bbox2.bottom());

    // If there's no intersection
    if x_right < x_left || y_bottom < y_top {
        return 0.0;
    }

    // Calculate intersection area
    let intersection_area = (x_right - x_left) * (y_bottom - y_top);

    // Calculate union area
    let bbox1_area = bbox1.area();
    let bbox2_area = bbox2.area();
    let union_area = bbox1_area + bbox2_area - intersection_area;

    // Avoid division by zero
    if union_area == 0.0 {
        return 0.0;
    }

    intersection_area / union_area
}

/// Calculate IoU matrix between two sets of bounding boxes.
///
/// # Arguments
///
/// * `bboxes1` - First set of bounding boxes
/// * `bboxes2` - Second set of bounding boxes
///
/// # Returns
///
/// Returns a 2D vector where `result[i][j]` is the IoU between `bboxes1[i]` and `bboxes2[j]`.
///
/// # Example
///
/// ```
/// use coco_eval::metrics::iou::calculate_iou_matrix;
/// use coco_eval::types::BoundingBox;
///
/// let bboxes1 = vec![BoundingBox::new(0.0, 0.0, 10.0, 10.0)];
/// let bboxes2 = vec![BoundingBox::new(5.0, 5.0, 10.0, 10.0)];
/// let iou_matrix = calculate_iou_matrix(&bboxes1, &bboxes2);
/// assert_eq!(iou_matrix.len(), 1);
/// assert_eq!(iou_matrix[0].len(), 1);
/// ```
pub fn calculate_iou_matrix(bboxes1: &[BoundingBox], bboxes2: &[BoundingBox]) -> Vec<Vec<f64>> {
    bboxes1
        .iter()
        .map(|bbox1| {
            bboxes2
                .iter()
                .map(|bbox2| calculate_iou(bbox1, bbox2))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_boxes() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let iou = calculate_iou(&bbox1, &bbox2);
        assert!((iou - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_overlap() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(20.0, 20.0, 10.0, 10.0);
        let iou = calculate_iou(&bbox1, &bbox2);
        assert_eq!(iou, 0.0);
    }

    #[test]
    fn test_partial_overlap() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(5.0, 5.0, 10.0, 10.0);
        let iou = calculate_iou(&bbox1, &bbox2);

        // Intersection: 5x5 = 25
        // Union: 100 + 100 - 25 = 175
        // IoU: 25/175 ≈ 0.1429
        assert!((iou - 0.142857).abs() < 1e-5);
    }

    #[test]
    fn test_iou_matrix() {
        let bboxes1 = vec![
            BoundingBox::new(0.0, 0.0, 10.0, 10.0),
            BoundingBox::new(5.0, 5.0, 10.0, 10.0),
        ];
        let bboxes2 = vec![
            BoundingBox::new(0.0, 0.0, 10.0, 10.0),
        ];

        let matrix = calculate_iou_matrix(&bboxes1, &bboxes2);
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 1);
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
    }
}
