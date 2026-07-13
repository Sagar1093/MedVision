import torch
from torchvision import models
from configs.config import *
from utils.dataloaders import get_dataloaders
from training.trainer import fit
from pathlib import Path


def main():

    print(f"Using Device: {DEVICE}")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    print("Train Images :", len(train_loader.dataset))
    print("Validation Images :", len(val_loader.dataset))

    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(weights=weights)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )
    model = model.to(DEVICE)

    print(model.fc)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE
    )
    checkpoint = CHECKPOINT_DIR/"resnet50_best.pth"
    checkpoint.parent.mkdir(parents=True,exist_ok=True)

    history = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler = scheduler,
        device=DEVICE,
        epochs=EPOCHS,
        checkpoint_path=checkpoint
    )
        
    

    

if __name__ == "__main__":
    main()