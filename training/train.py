import torch

from configs.config import *
from utils.dataloaders import get_dataloaders
from training.trainer import fit

from utils.class_weights import get_class_weights
from models.model_factory import get_model


def main():

    print(f"Using Device: {DEVICE}")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    print("Train Images :", len(train_loader.dataset))
    print("Validation Images :", len(val_loader.dataset))

   

    model = get_model(
        MODEL_NAME,
        NUM_CLASSES
    ).to(DEVICE)

    class_weights = get_class_weights(
        train_loader.dataset,
        DEVICE
    )
    print("Class Weights:", class_weights)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

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
    checkpoint = checkpoint = CHECKPOINT_DIR / f"{MODEL_NAME}_best.pth"
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