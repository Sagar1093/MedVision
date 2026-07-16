import torch
from models.model_factory import get_model
from configs.config import *
from backend.gradcam import Gradcam_genarator,image_to_base64

class Predictor:
    def __init__(self):
        self.model = get_model(
            MODEL_NAME,
            NUM_CLASSES
        ).to(DEVICE)

        checkpoint = torch.load(
            CHECKPOINT_DIR/f"{MODEL_NAME}_best.pth",
            map_location=DEVICE
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.gradcam = Gradcam_genarator(self.model)


        self.classes = [
            "Normal",
            "Pneumonia",
            "Tuberculosis"
        ]

    
    def predict(self,image_tensor,rgb_image):
        image_tensor = image_tensor.to(DEVICE)

        outputs = self.model(image_tensor)

        heatmap = self.gradcam.generate(
            image_tensor,
            rgb_image
        )

        heatmap_base64 = image_to_base64(heatmap)

        probabilites = torch.softmax(
            outputs,
            dim=1
        )
        confidence, prediction = torch.max(
            probabilites,
            dim=1
        )
        return {
            "prediction":self.classes[prediction.item()],
            "confidence": round(confidence.item() * 100,2),
            "probabilities": {
                self.classes[i]: round(probabilites[0][i].item() * 100,2)
                for i in range(NUM_CLASSES)
            },
            "heatmap":heatmap_base64
        }