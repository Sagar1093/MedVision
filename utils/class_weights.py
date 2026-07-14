import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def get_class_weights(dataset, device):
    """
    Compute class weights from an ImageFolder dataset.
    """

    labels = np.array(dataset.targets)

   

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )