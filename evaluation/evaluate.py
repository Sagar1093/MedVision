import torch
import numpy as np
from tqdm import tqdm


def evaluate_model(
        model,
        dataloader,
        device
):
    model.eval()
    predictions = []
    labels_list = []
    probabilities = []

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc="Testing"
        )
        for images,labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs,dim=1)
            _,preds = torch.max(outputs,1)
            predictions.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    return (
        np.array(labels_list),
        np.array(predictions),
        np.array(probabilities)
    )