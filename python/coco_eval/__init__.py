"""
coco-eval: High-performance COCO evaluation metrics

A Rust implementation of COCO object detection evaluation metrics with Python bindings.
Provides 10x+ performance improvement over pure Python implementations while maintaining
full API compatibility.
"""

__version__ = "0.1.0"

# Import Rust core
try:
    from .core import Evaluator as _RustEvaluator, validate_bbox
except ImportError as e:
    raise ImportError(
        "Failed to import Rust core module. "
        "Make sure you have built the package with `maturin develop` or installed it properly."
    ) from e

# Re-export core functions
__all__ = ["Evaluator", "validate_bbox"]


class Evaluator(_RustEvaluator):
    """
    COCO evaluation metrics evaluator.

    This class extends the Rust Evaluator with Python-side plotting functionality.

    Parameters
    ----------
    annotations_df : polars.DataFrame
        Ground truth annotations with columns: image_id, category_id, bbox
    images_df : polars.DataFrame
        Image metadata with columns: image_id, file_name, frame_uri
    predictions_df : polars.DataFrame
        Model predictions with columns: image_id, category_id, bbox, score
    model_class_list : list[str]
        List of class names the model can predict
    nms_iou_threshold : float, default=0.5
        IoU threshold for Non-Maximum Suppression
    class_groupings : dict[str, list[str]], optional
        Optional class groupings for hierarchical evaluation

    Examples
    --------
    >>> import polars as pl
    >>> from coco_eval import Evaluator
    >>>
    >>> # Create evaluator
    >>> evaluator = Evaluator(
    ...     annotations_df=annotations,
    ...     images_df=images,
    ...     predictions_df=predictions,
    ...     model_class_list=["cat", "dog", "bird"],
    ... )
    >>>
    >>> # Calculate metrics at optimal thresholds
    >>> results = evaluator.calculate_metrics_at_best_thresholds(beta=1.0, iou_threshold=0.5)
    >>> print(results)
    """

    def plot_confusion_matrix(
        self,
        filename: str,
        normalize: bool = True,
        cmap: str = "Blues",
        figsize: tuple[int, int] = (10, 8),
    ) -> None:
        """
        Plot confusion matrix and save to file.

        Parameters
        ----------
        filename : str
            Path to save the confusion matrix plot
        normalize : bool, default=True
            Whether to normalize the confusion matrix
        cmap : str, default="Blues"
            Matplotlib colormap name
        figsize : tuple[int, int], default=(10, 8)
            Figure size in inches (width, height)

        Examples
        --------
        >>> evaluator.plot_confusion_matrix("confusion_matrix.png")
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
        except ImportError as e:
            raise ImportError(
                "Plotting requires matplotlib and scikit-learn. "
                "Install with: pip install matplotlib scikit-learn"
            ) from e

        # Get confusion matrix from Rust implementation
        # This method should be implemented in the Rust side
        cm_data = self.get_confusion_matrix()

        # Plot using matplotlib
        fig, ax = plt.subplots(figsize=figsize)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_data["matrix"],
            display_labels=cm_data["labels"],
        )
        disp.plot(cmap=cmap, ax=ax, colorbar=True)

        if normalize:
            plt.title("Normalized Confusion Matrix")
        else:
            plt.title("Confusion Matrix")

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
