from pathlib import Path
from torchvision import datasets
from torch.utils.data import DataLoader
from utils.transforms import train_transform, test_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "raw" / "archive"

def get_dataloaders(batch_size=32, num_workers=4):
    train_dataset = datasets.ImageFolder(
        DATASET_ROOT / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        DATASET_ROOT / "val",
        transform=test_transform
    )

    test_dataset = datasets.ImageFolder(
        DATASET_ROOT / "test",
        transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader