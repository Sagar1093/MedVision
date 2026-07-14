from torchvision import transforms

train_transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.02,0.02),
            scale=(0.95,1.05)
        ),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.2,
            scale=(0.02,0.08)
        ),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
)

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
])