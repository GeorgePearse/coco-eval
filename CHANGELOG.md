# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-11-01

### Added
- Initial release of coco-eval
- Core data structures for COCO format (BoundingBox, Annotation, Category, Image, CocoDataset)
- JSON loading and parsing with validation
- IoU (Intersection over Union) calculation
- Precision and Recall calculation
- F1 Score calculation
- Average Precision (AP) calculation with 101-point interpolation
- mean Average Precision (mAP) calculation
- Confidence score thresholding
- Threshold range generation utilities
- Comprehensive error handling with custom error types
- Full test coverage for core functionality
- Documentation and usage examples

### Features by Module
- **loader**: Load COCO JSON from files or strings with validation
- **types**: Type-safe COCO format data structures
- **metrics::iou**: Calculate IoU between bounding boxes
- **metrics::ap**: Average Precision and mAP calculation
- **metrics::precision_recall**: Precision and recall with interpolation
- **metrics::f1_score**: F1 score with optimal threshold finding
- **threshold**: Filter annotations by confidence scores
- **evaluator**: Main evaluation orchestrator (placeholder for full implementation)

### Documentation
- Comprehensive README with examples
- API documentation for all public functions
- Usage examples in doc comments
- COCO format specification

[Unreleased]: https://github.com/GeorgePearse/coco-eval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/GeorgePearse/coco-eval/releases/tag/v0.1.0
