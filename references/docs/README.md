This package provides comprehensive evaluation metrics for machine learning models, particularly focused on object detection tasks.

## Main API Methods

### 1. Evaluate at Current Thresholds

```python
evaluation_results = evaluator.calculate_metrics_at_current_thresholds()
```

Calculates F1 score, precision, and recall using the existing confidence scores in your predictions.

### 2. Find Optimal Thresholds

```python
evaluation_results = evaluator.calculate_metrics_at_best_thresholds(beta=1)
```

Performs a grid search to find the optimal confidence threshold for each class that maximizes the F-beta score. The beta parameter controls the precision/recall trade-off:

- `beta < 1`: Favors precision
- `beta = 1`: Balanced F1 score
- `beta > 1`: Favors recall

## Usage Example

```python
from evaluation.eval_metrics import Evaluator

evaluator = Evaluator(
    annotations_df=annotations,
    images_df=images,
    predictions_df=predictions,
    model_class_list=["class1", "class2"],
    nms_iou_threshold=0.5
)

# Get metrics at optimal thresholds
results = evaluator.calculate_metrics_at_best_thresholds(beta=1)
```
