import torch
from torchvision import models


def get_model(model_name, num_classes):

    model_name = model_name.lower()

    if model_name == "resnet50":

        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        model.fc = torch.nn.Linear(
            model.fc.in_features,
            num_classes
        )

    elif model_name == "densenet121":

        model = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        model.classifier = torch.nn.Linear(
            model.classifier.in_features,
            num_classes
        )

    elif model_name == "efficientnet_b3":

        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT
        )

        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model