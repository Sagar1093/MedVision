from PIL import Image
from io import BytesIO
from utils.transforms import test_transform

def preprocess_image(image_bytes):
    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image = test_transform(image)
    image = image.unsqueeze(0)

    return image