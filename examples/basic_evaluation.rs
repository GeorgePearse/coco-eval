//! Basic evaluation example demonstrating core functionality.

use coco_eval::{
    load_from_string, evaluator::evaluate, metrics::iou::calculate_iou, BoundingBox,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== COCO Evaluation Example ===\n");

    // Example 1: IoU Calculation
    println!("1. IoU Calculation");
    let bbox1 = BoundingBox::new(10.0, 10.0, 50.0, 50.0);
    let bbox2 = BoundingBox::new(30.0, 30.0, 50.0, 50.0);
    let iou = calculate_iou(&bbox1, &bbox2);
    println!("   IoU between overlapping boxes: {:.4}", iou);
    println!();

    // Example 2: Load COCO annotations
    println!("2. Loading COCO Annotations");
    let ground_truth_json = r#"{
        "images": [
            {
                "id": 1,
                "file_name": "image1.jpg",
                "height": 480,
                "width": 640
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100.0, 100.0, 200.0, 150.0],
                "area": 30000.0
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [350.0, 200.0, 100.0, 120.0],
                "area": 12000.0
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "human"
            },
            {
                "id": 2,
                "name": "car",
                "supercategory": "vehicle"
            }
        ]
    }"#;

    let ground_truth = load_from_string(ground_truth_json)?;
    println!("   Loaded {} ground truth annotations", ground_truth.annotations.len());
    println!("   Categories: {:?}", ground_truth.categories.iter().map(|c| &c.name).collect::<Vec<_>>());
    println!();

    // Example 3: Load predictions
    println!("3. Loading Predictions");
    let predictions_json = r#"{
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [105.0, 98.0, 195.0, 155.0],
                "score": 0.95
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [348.0, 198.0, 105.0, 125.0],
                "score": 0.87
            },
            {
                "id": 3,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50.0, 50.0, 80.0, 90.0],
                "score": 0.42
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person"
            },
            {
                "id": 2,
                "name": "car"
            }
        ]
    }"#;

    let predictions = load_from_string(predictions_json)?;
    println!("   Loaded {} predictions", predictions.annotations.len());
    println!();

    // Example 4: Confidence filtering
    println!("4. Confidence Thresholding");
    use coco_eval::filter_dataset_by_confidence;

    let high_conf_preds = filter_dataset_by_confidence(&predictions, 0.7)?;
    println!("   Predictions with confidence >= 0.7: {}", high_conf_preds.annotations.len());

    let medium_conf_preds = filter_dataset_by_confidence(&predictions, 0.5)?;
    println!("   Predictions with confidence >= 0.5: {}", medium_conf_preds.annotations.len());
    println!();

    // Example 5: Evaluation
    println!("5. Running Full Evaluation");
    let metrics = evaluate(&ground_truth, &predictions, None, None)?;
    println!("   Evaluation complete!");
    println!();
    println!("   Overall Metrics:");
    println!("   ├─ mAP (all IoU thresholds): {:.4}", metrics.map);
    println!("   ├─ AP50 (IoU=0.50): {:.4}", metrics.ap50);
    println!("   └─ AP75 (IoU=0.75): {:.4}", metrics.ap75);
    println!();
    println!("   Per-Class AP:");
    for (cat_id, ap) in &metrics.ap_per_class {
        let cat_name = ground_truth.categories.iter()
            .find(|c| c.id == *cat_id)
            .map(|c| c.name.as_str())
            .unwrap_or("unknown");
        println!("   ├─ {} (id={}): {:.4}", cat_name, cat_id, ap);
    }
    println!();

    // Example 6: Confidence Threshold Analysis
    println!("6. Precision/Recall/F1 at Confidence Thresholds");
    println!("   Threshold | Precision | Recall | F1 Score");
    println!("   ----------|-----------|--------|----------");

    for i in 0..metrics.precision_at_thresholds.len().min(5) {
        let (threshold, precision) = metrics.precision_at_thresholds[i];
        let (_, recall) = metrics.recall_at_thresholds[i];
        let (_, f1) = metrics.f1_at_thresholds[i];
        println!("   {:>8.2} | {:>9.4} | {:>6.4} | {:>8.4}",
            threshold, precision, recall, f1);
    }
    println!("   ... ({} thresholds total)", metrics.precision_at_thresholds.len());
    println!();

    // Example 7: Manual Precision/Recall calculation
    println!("7. Computing Precision and Recall (Manual Example)");
    use coco_eval::metrics::precision_recall::calculate_precision_recall;
    use coco_eval::metrics::f1_score::calculate_f1_from_pr;

    let pr = calculate_precision_recall(8, 2, 3);
    println!("   For TP=8, FP=2, FN=3:");
    println!("   ├─ Precision: {:.4} ({}/ {})", pr.precision, pr.true_positives, pr.true_positives + pr.false_positives);
    println!("   ├─ Recall: {:.4} ({}/{})", pr.recall, pr.true_positives, pr.true_positives + pr.false_negatives);

    let f1 = calculate_f1_from_pr(&pr);
    println!("   └─ F1 Score: {:.4}", f1);
    println!();

    println!("=== Example Complete ===");

    Ok(())
}
