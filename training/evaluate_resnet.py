import torch
from torchvision import models

from configs.config import *
from utils.dataloaders import get_dataloaders
from evaluation.evaluate import evaluate_model
from evaluation.metrics import calculate_metrics
from evaluation.confusion_matrix import plot_confusion_matrix

def main():
    from pathlib import Path

    results_dir = Path("results/resnet50")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using Device: {DEVICE}")
    _, _, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )
    model = models.resnet50(
        weights=None
    )

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_DIR / "resnet50_best.pth",
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    y_true, y_pred, y_prob = evaluate_model(
        model,
        test_loader,
        DEVICE,
    )
    metrics = calculate_metrics(
        y_true,
        y_pred,
    )
    print("\n========== TEST RESULTS ==========\n")

    print(f"Accuracy : {metrics['accuracy']:.4f}")

    print(f"Precision : {metrics['precision']:.4f}")

    print(f"Recall : {metrics['recall']:.4f}")

    print(f"F1 Score : {metrics['f1']:.4f}")

    print("\nClassification Report\n")

    print(metrics["report"])

    plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=test_loader.dataset.classes,
    save_path="results/resnet50/confusion_matrix.png",
)
if __name__ == "__main__":
    main()