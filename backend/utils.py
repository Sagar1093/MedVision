from PIL import Image
from io import BytesIO
import numpy as np
from utils.transforms import test_transform

def preprocess_image(image_bytes):
    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    rgb_image = np.array(image.resize((224,224))).astype(np.float32)/255.0

    image = test_transform(image)
    image = image.unsqueeze(0)

    return image,rgb_image