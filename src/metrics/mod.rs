//! Metrics calculation modules for COCO evaluation.

pub mod iou;
pub mod ap;
pub mod precision_recall;
pub mod f1_score;
pub mod fbeta;

pub use iou::calculate_iou;
pub use ap::{calculate_ap, calculate_map};
pub use precision_recall::{calculate_precision_recall, PrecisionRecall};
pub use f1_score::calculate_f1_score;
pub use fbeta::{
    FBetaResult, calculate_fbeta, calculate_f1, calculate_precision, calculate_recall,
    calculate_fbeta_from_counts, generate_threshold_range, find_best_threshold,
};
