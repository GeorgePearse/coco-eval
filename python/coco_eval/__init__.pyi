"""Type stubs for coco-eval"""

from typing import Optional
import polars as pl
import numpy as np
import numpy.typing as npt

__version__: str

class Evaluator:
    """COCO evaluation metrics evaluator."""

    def __init__(
        self,
        annotations_df: pl.DataFrame,
        images_df: pl.DataFrame,
        predictions_df: pl.DataFrame,
        model_class_list: list[str],
        nms_iou_threshold: float = 0.5,
        class_groupings: Optional[dict[str, list[str]]] = None,
    ) -> None: ...

    def calculate_metrics_at_best_thresholds(
        self,
        beta: float = 1.0,
        iou_threshold: float = 0.5,
    ) -> pl.DataFrame:
        """
        Calculate metrics using optimal confidence thresholds for each class.

        Parameters
        ----------
        beta : float, default=1.0
            F-beta score beta parameter (1.0 = F1 score)
        iou_threshold : float, default=0.5
            IoU threshold for considering a detection as correct

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: class_id, class_name, iou_thr, best_fbeta,
            conf, precision, recall, tp, fp, fn
        """
        ...

    def calculate_metrics_at_current_thresholds(
        self,
        beta: float = 1.0,
        iou_thr: float = 0.5,
    ) -> pl.DataFrame:
        """
        Calculate metrics at current confidence thresholds.

        Parameters
        ----------
        beta : float, default=1.0
            F-beta score beta parameter (1.0 = F1 score)
        iou_thr : float, default=0.5
            IoU threshold for considering a detection as correct

        Returns
        -------
        pl.DataFrame
            DataFrame with metrics for each class
        """
        ...

    def summarize_camera_statistics(
        self,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Summarize statistics per camera.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame]
            - Camera-level statistics (aggregate across classes)
            - Camera-class-level statistics (per camera and class)
        """
        ...

    def get_confusion_matrix(self) -> dict[str, npt.NDArray[np.int64]]:
        """
        Get confusion matrix data.

        Returns
        -------
        dict
            Dictionary with keys 'matrix' (NxN array) and 'labels' (list of class names)
        """
        ...

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
        """
        ...

def validate_bbox(bbox: list[float]) -> bool:
    """
    Validate a bounding box.

    Parameters
    ----------
    bbox : list[float]
        Bounding box in format [x, y, width, height]

    Returns
    -------
    bool
        True if bbox is valid, False otherwise
    """
    ...

__all__ = ["Evaluator", "validate_bbox"]
