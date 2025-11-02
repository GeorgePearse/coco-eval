use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use coco_eval::metrics::{
    calculate_iou, calculate_ap, calculate_precision, calculate_recall,
    calculate_f1, calculate_fbeta, generate_threshold_range,
};
use coco_eval::nms::{Detection, non_maximum_suppression};
use coco_eval::types::BoundingBox;

fn bench_iou_calculation(c: &mut Criterion) {
    let bbox1 = BoundingBox::new(10.0, 10.0, 50.0, 50.0);
    let bbox2 = BoundingBox::new(30.0, 30.0, 50.0, 50.0);

    c.bench_function("iou_single", |b| {
        b.iter(|| {
            calculate_iou(black_box(&bbox1), black_box(&bbox2))
        });
    });
}

fn bench_iou_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("iou_matrix");

    for size in [10, 50, 100, 500].iter() {
        let boxes: Vec<BoundingBox> = (0..*size)
            .map(|i| {
                let offset = (i as f64) * 2.0;
                BoundingBox::new(offset, offset, 50.0, 50.0)
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                for i in 0..boxes.len() {
                    for j in 0..boxes.len() {
                        black_box(calculate_iou(&boxes[i], &boxes[j]));
                    }
                }
            });
        });
    }
    group.finish();
}

fn bench_nms(c: &mut Criterion) {
    let mut group = c.benchmark_group("nms");

    for num_boxes in [10, 50, 100, 500].iter() {
        let detections: Vec<Detection> = (0..*num_boxes)
            .map(|i| {
                let offset = (i as f64) * 10.0;
                Detection {
                    bbox: [offset, offset, offset + 50.0, offset + 50.0],
                    score: 0.9 - (i as f64) * 0.001,
                    index: i,
                }
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(num_boxes), num_boxes, |b, _| {
            b.iter(|| {
                non_maximum_suppression(black_box(&detections), black_box(0.5))
            });
        });
    }
    group.finish();
}

fn bench_ap_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ap_calculation");

    for num_detections in [10, 50, 100, 500].iter() {
        let precision: Vec<f64> = (0..*num_detections)
            .map(|i| 1.0 - (i as f64) / (*num_detections as f64))
            .collect();
        let recall: Vec<f64> = (0..*num_detections)
            .map(|i| (i as f64) / (*num_detections as f64))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(num_detections), num_detections, |b, _| {
            b.iter(|| {
                calculate_ap(black_box(&precision), black_box(&recall))
            });
        });
    }
    group.finish();
}

fn bench_fbeta_metrics(c: &mut Criterion) {
    c.bench_function("precision_calculation", |b| {
        b.iter(|| {
            calculate_precision(black_box(80), black_box(20))
        });
    });

    c.bench_function("recall_calculation", |b| {
        b.iter(|| {
            calculate_recall(black_box(80), black_box(20))
        });
    });

    c.bench_function("f1_calculation", |b| {
        b.iter(|| {
            calculate_f1(black_box(0.8), black_box(0.7))
        });
    });

    c.bench_function("fbeta_calculation", |b| {
        b.iter(|| {
            calculate_fbeta(black_box(0.8), black_box(0.7), black_box(2.0))
        });
    });
}

fn bench_threshold_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold_generation");

    for num_thresholds in [11, 51, 101, 501].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(num_thresholds), num_thresholds, |b, &n| {
            b.iter(|| {
                generate_threshold_range(black_box(n))
            });
        });
    }
    group.finish();
}

fn bench_nms_overlapping(c: &mut Criterion) {
    // Benchmark NMS with heavily overlapping boxes
    let detections: Vec<Detection> = (0..100)
        .map(|i| {
            Detection {
                bbox: [10.0 + (i as f64), 10.0 + (i as f64), 60.0, 60.0],
                score: 0.9 - (i as f64) * 0.005,
                index: i,
            }
        })
        .collect();

    c.bench_function("nms_overlapping_100", |b| {
        b.iter(|| {
            non_maximum_suppression(black_box(&detections), black_box(0.5))
        });
    });
}

criterion_group!(
    benches,
    bench_iou_calculation,
    bench_iou_matrix,
    bench_nms,
    bench_ap_calculation,
    bench_fbeta_metrics,
    bench_threshold_generation,
    bench_nms_overlapping,
);
criterion_main!(benches);
