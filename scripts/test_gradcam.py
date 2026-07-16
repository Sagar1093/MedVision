from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from models.model_factory import get_model
from configs.config import *
from backend.gradcam import Gradcam_genarator
from utils.transforms import test_transform


MODEL_PATH = CHECKPOINT_DIR / f"{MODEL_NAME}_best.pth"

IMAGE_PATH = Path("datasets/raw/archive/test/pneumonia/pneumonia-75.jpg")
# Replace with any image from your test set


def main():

    model = get_model(
        MODEL_NAME,
        NUM_CLASSES
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    image = Image.open(IMAGE_PATH).convert("RGB")

    rgb_image = np.array(
        image.resize((224, 224))
    ).astype(np.float32) / 255.0

    image_tensor = test_transform(image).unsqueeze(0)

    gradcam = Gradcam_genarator(model)

    heatmap = gradcam.generate(
        image_tensor.to(DEVICE),
        rgb_image
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(
        str(output_dir / "gradcam_test.png"),
        cv2.cvtColor(
            heatmap,
            cv2.COLOR_RGB2BGR
        )
    )

    print("Grad-CAM saved successfully!")


if __name__ == "__main__":
    main()