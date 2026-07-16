import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models.resnet import ResNet
from torchvision.models.densenet import DenseNet
from torchvision.models.efficientnet import EfficientNet

def image_to_base64(image):
    _,buffer = cv2.imencode(".png",image)
    return base64.b64encode(
        buffer.tobytes()
    ).decode("utf-8")

class Gradcam_genarator:
    def __init__(self,model):
         if isinstance(model, ResNet):
            target_layers = [model.layer4[-1]]

         elif isinstance(model, DenseNet):
            target_layers = [model.features[-1]]

         elif isinstance(model, EfficientNet):
            target_layers = [model.features[-1]]

         else:
            raise ValueError(f"Unsupported model: {type(model).__name__}")

         self.cam = GradCAM(
              model=model,
              target_layers=target_layers
         )

    def generate(self,image_tensor,rgb_image):
         image_tensor = image_tensor.requires_grad_(True)
         grayscale_cam = self.cam(
              input_tensor=image_tensor
         )[0]

         visualization = show_cam_on_image(
              rgb_image,
              grayscale_cam,
              use_rgb=True
         )

         return visualization