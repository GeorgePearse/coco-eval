//! Comprehensive tests for loader and type modules.
//!
//! Tests for BoundingBox, Annotation, CocoDataset loading, and type conversions.

use coco_eval::types::{Annotation, BoundingBox, Category, CocoDataset, Image};

#[test]
fn test_bounding_box_new() {
    let bbox = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
    assert_eq!(bbox.x, 10.0);
    assert_eq!(bbox.y, 20.0);
    assert_eq!(bbox.width, 30.0);
    assert_eq!(bbox.height, 40.0);
}

#[test]
fn test_bounding_box_area() {
    let bbox = BoundingBox::new(0.0, 0.0, 10.0, 5.0);
    assert_eq!(bbox.area(), 50.0);
}

#[test]
fn test_bounding_box_area_zero() {
    let bbox = BoundingBox::new(0.0, 0.0, 0.0, 0.0);
    assert_eq!(bbox.area(), 0.0);
}

#[test]
fn test_bounding_box_negative_dimensions() {
    // Negative width/height should still calculate area correctly
    let bbox = BoundingBox::new(10.0, 10.0, -5.0, -5.0);
    assert_eq!(bbox.area(), 25.0);
}

#[test]
fn test_bounding_box_right() {
    let bbox = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
    assert_eq!(bbox.right(), 40.0); // x + width
}

#[test]
fn test_bounding_box_bottom() {
    let bbox = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
    assert_eq!(bbox.bottom(), 60.0); // y + height
}

#[test]
fn test_bounding_box_is_valid_positive() {
    let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
    assert!(bbox.is_valid());
}

#[test]
fn test_bounding_box_is_valid_zero_width() {
    let bbox = BoundingBox::new(0.0, 0.0, 0.0, 10.0);
    assert!(!bbox.is_valid());
}

#[test]
fn test_bounding_box_is_valid_zero_height() {
    let bbox = BoundingBox::new(0.0, 0.0, 10.0, 0.0);
    assert!(!bbox.is_valid());
}

#[test]
fn test_bounding_box_is_valid_negative_width() {
    let bbox = BoundingBox::new(0.0, 0.0, -10.0, 10.0);
    assert!(!bbox.is_valid());
}

#[test]
fn test_bounding_box_is_valid_negative_height() {
    let bbox = BoundingBox::new(0.0, 0.0, 10.0, -10.0);
    assert!(!bbox.is_valid());
}

#[test]
fn test_bounding_box_large_coordinates() {
    let bbox = BoundingBox::new(1e6, 1e6, 1e3, 1e3);
    assert!(bbox.is_valid());
    assert_eq!(bbox.area(), 1e6);
}

#[test]
fn test_bounding_box_very_small_dimensions() {
    let bbox = BoundingBox::new(0.0, 0.0, 1e-10, 1e-10);
    assert!(bbox.is_valid());
    assert!(bbox.area() > 0.0);
}

#[test]
fn test_annotation_to_bbox_valid() {
    let ann = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0, 40.0],
        area: Some(1200.0),
        iscrowd: None,
        score: None,
    };

    let bbox = ann.to_bbox().unwrap();
    assert_eq!(bbox.x, 10.0);
    assert_eq!(bbox.y, 20.0);
    assert_eq!(bbox.width, 30.0);
    assert_eq!(bbox.height, 40.0);
}

#[test]
fn test_annotation_to_bbox_invalid_length() {
    let ann = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0], // Only 3 values
        area: None,
        iscrowd: None,
        score: None,
    };

    let result = ann.to_bbox();
    assert!(result.is_err());
}

#[test]
fn test_annotation_to_bbox_empty() {
    let ann = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: vec![],
        area: None,
        iscrowd: None,
        score: None,
    };

    let result = ann.to_bbox();
    assert!(result.is_err());
}

#[test]
fn test_annotation_to_bbox_too_many_values() {
    let ann = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0, 40.0, 50.0], // 5 values
        area: None,
        iscrowd: None,
        score: None,
    };

    let result = ann.to_bbox();
    assert!(result.is_err());
}

#[test]
fn test_annotation_confidence_with_score() {
    let ann = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0, 40.0],
        area: None,
        iscrowd: None,
        score: Some(0.85),
    };

    assert_eq!(ann.confidence(), 0.85);
}

#[test]
fn test_annotation_confidence_without_score() {
    let ann = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0, 40.0],
        area: None,
        iscrowd: None,
        score: None,
    };

    // Default confidence should be 1.0
    assert_eq!(ann.confidence(), 1.0);
}

#[test]
fn test_annotation_confidence_zero() {
    let ann = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0, 40.0],
        area: None,
        iscrowd: None,
        score: Some(0.0),
    };

    assert_eq!(ann.confidence(), 0.0);
}

#[test]
fn test_annotation_confidence_one() {
    let ann = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0, 40.0],
        area: None,
        iscrowd: None,
        score: Some(1.0),
    };

    assert_eq!(ann.confidence(), 1.0);
}

#[test]
fn test_coco_dataset_empty() {
    let dataset = CocoDataset {
        images: None,
        annotations: vec![],
        categories: vec![],
    };

    assert_eq!(dataset.annotations.len(), 0);
    assert_eq!(dataset.categories.len(), 0);
}

#[test]
fn test_coco_dataset_with_images() {
    let dataset = CocoDataset {
        images: Some(vec![Image {
            id: 1,
            file_name: "test.jpg".to_string(),
            width: 640,
            height: 480,
        }]),
        annotations: vec![],
        categories: vec![],
    };

    assert!(dataset.images.is_some());
    assert_eq!(dataset.images.as_ref().unwrap().len(), 1);
}

#[test]
fn test_coco_dataset_multiple_categories() {
    let dataset = CocoDataset {
        images: None,
        annotations: vec![],
        categories: vec![
            Category {
                id: 1,
                name: "person".to_string(),
                supercategory: None,
            },
            Category {
                id: 2,
                name: "car".to_string(),
                supercategory: Some("vehicle".to_string()),
            },
            Category {
                id: 3,
                name: "dog".to_string(),
                supercategory: Some("animal".to_string()),
            },
        ],
    };

    assert_eq!(dataset.categories.len(), 3);
    assert_eq!(dataset.categories[0].name, "person");
    assert_eq!(dataset.categories[1].name, "car");
    assert_eq!(dataset.categories[2].name, "dog");
}

#[test]
fn test_category_without_supercategory() {
    let cat = Category {
        id: 1,
        name: "person".to_string(),
        supercategory: None,
    };

    assert_eq!(cat.id, 1);
    assert_eq!(cat.name, "person");
    assert!(cat.supercategory.is_none());
}

#[test]
fn test_category_with_supercategory() {
    let cat = Category {
        id: 2,
        name: "car".to_string(),
        supercategory: Some("vehicle".to_string()),
    };

    assert_eq!(cat.id, 2);
    assert_eq!(cat.name, "car");
    assert_eq!(cat.supercategory, Some("vehicle".to_string()));
}

#[test]
fn test_image_with_dimensions() {
    let img = Image {
        id: 1,
        file_name: "test.jpg".to_string(),
        width: 1920,
        height: 1080,
    };

    assert_eq!(img.id, 1);
    assert_eq!(img.file_name, "test.jpg");
    assert_eq!(img.width, 1920);
    assert_eq!(img.height, 1080);
}

#[test]
fn test_image_dimensions_zero() {
    // Test with zero dimensions (edge case)
    let img = Image {
        id: 2,
        file_name: "unknown.jpg".to_string(),
        width: 0,
        height: 0,
    };

    assert_eq!(img.id, 2);
    assert_eq!(img.width, 0);
    assert_eq!(img.height, 0);
}

#[test]
fn test_annotation_clone() {
    let ann1 = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0, 40.0],
        area: Some(1200.0),
        iscrowd: Some(0),
        score: Some(0.95),
    };

    let ann2 = ann1.clone();
    assert_eq!(ann1.id, ann2.id);
    assert_eq!(ann1.image_id, ann2.image_id);
    assert_eq!(ann1.category_id, ann2.category_id);
    assert_eq!(ann1.bbox, ann2.bbox);
    assert_eq!(ann1.area, ann2.area);
    assert_eq!(ann1.score, ann2.score);
}

#[test]
fn test_annotation_iscrowd_flag() {
    let crowd = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0, 40.0],
        area: None,
        iscrowd: Some(1),
        score: None,
    };

    assert_eq!(crowd.iscrowd, Some(1));

    let not_crowd = Annotation {
        id: 2,
        image_id: 1,
        category_id: 1,
        bbox: vec![50.0, 60.0, 70.0, 80.0],
        area: None,
        iscrowd: Some(0),
        score: None,
    };

    assert_eq!(not_crowd.iscrowd, Some(0));
}

#[test]
fn test_bounding_box_equality() {
    let bbox1 = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
    let bbox2 = BoundingBox::new(10.0, 20.0, 30.0, 40.0);
    let bbox3 = BoundingBox::new(10.0, 20.0, 30.0, 41.0);

    // Note: BoundingBox doesn't derive PartialEq, so we compare fields manually
    assert_eq!(bbox1.x, bbox2.x);
    assert_eq!(bbox1.y, bbox2.y);
    assert_eq!(bbox1.width, bbox2.width);
    assert_eq!(bbox1.height, bbox2.height);

    assert_ne!(bbox1.height, bbox3.height);
}

#[test]
fn test_annotation_with_all_fields() {
    let ann = Annotation {
        id: 42,
        image_id: 100,
        category_id: 5,
        bbox: vec![15.5, 25.5, 100.0, 200.0],
        area: Some(20000.0),
        iscrowd: Some(0),
        score: Some(0.999),
    };

    assert_eq!(ann.id, 42);
    assert_eq!(ann.image_id, 100);
    assert_eq!(ann.category_id, 5);
    assert_eq!(ann.bbox.len(), 4);
    assert_eq!(ann.area, Some(20000.0));
    assert_eq!(ann.iscrowd, Some(0));
    assert_eq!(ann.confidence(), 0.999);
}

#[test]
fn test_coco_dataset_realistic() {
    let dataset = CocoDataset {
        images: Some(vec![
            Image {
                id: 1,
                file_name: "image1.jpg".to_string(),
                width: 640,
                height: 480,
            },
            Image {
                id: 2,
                file_name: "image2.jpg".to_string(),
                width: 1920,
                height: 1080,
            },
        ]),
        annotations: vec![
            Annotation {
                id: 1,
                image_id: 1,
                category_id: 1,
                bbox: vec![100.0, 100.0, 50.0, 50.0],
                area: Some(2500.0),
                iscrowd: Some(0),
                score: Some(0.95),
            },
            Annotation {
                id: 2,
                image_id: 1,
                category_id: 2,
                bbox: vec![200.0, 200.0, 100.0, 100.0],
                area: Some(10000.0),
                iscrowd: Some(0),
                score: Some(0.88),
            },
            Annotation {
                id: 3,
                image_id: 2,
                category_id: 1,
                bbox: vec![500.0, 500.0, 200.0, 200.0],
                area: Some(40000.0),
                iscrowd: Some(0),
                score: Some(0.92),
            },
        ],
        categories: vec![
            Category {
                id: 1,
                name: "person".to_string(),
                supercategory: None,
            },
            Category {
                id: 2,
                name: "car".to_string(),
                supercategory: Some("vehicle".to_string()),
            },
        ],
    };

    assert_eq!(dataset.images.as_ref().unwrap().len(), 2);
    assert_eq!(dataset.annotations.len(), 3);
    assert_eq!(dataset.categories.len(), 2);

    // Verify annotations per image
    let img1_anns: Vec<_> = dataset
        .annotations
        .iter()
        .filter(|a| a.image_id == 1)
        .collect();
    assert_eq!(img1_anns.len(), 2);

    let img2_anns: Vec<_> = dataset
        .annotations
        .iter()
        .filter(|a| a.image_id == 2)
        .collect();
    assert_eq!(img2_anns.len(), 1);
}

#[test]
fn test_bounding_box_floating_point_precision() {
    // Test with values that might have floating point precision issues
    let bbox = BoundingBox::new(0.1 + 0.2, 0.3, 0.4, 0.5);
    // 0.1 + 0.2 = 0.30000000000000004 in floating point
    assert!((bbox.x - 0.3).abs() < 1e-10);
}

#[test]
fn test_annotation_score_edge_values() {
    // Test with score exactly at boundaries
    let ann_low = Annotation {
        id: 1,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0, 40.0],
        area: None,
        iscrowd: None,
        score: Some(0.0),
    };
    assert_eq!(ann_low.confidence(), 0.0);

    let ann_high = Annotation {
        id: 2,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0, 40.0],
        area: None,
        iscrowd: None,
        score: Some(1.0),
    };
    assert_eq!(ann_high.confidence(), 1.0);

    // Test with score outside normal range (though shouldn't happen in practice)
    let ann_over = Annotation {
        id: 3,
        image_id: 1,
        category_id: 1,
        bbox: vec![10.0, 20.0, 30.0, 40.0],
        area: None,
        iscrowd: None,
        score: Some(1.5),
    };
    assert_eq!(ann_over.confidence(), 1.5);
}
