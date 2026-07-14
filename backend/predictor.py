import torch
from models.model_factory import get_model
from configs.config import *

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

        self.classes = [
            "Normal",
            "Pneumonia",
            "Tuberculosis"
        ]

    @torch.no_grad()
    def predict(self,image_tensor):
        image_tensor = image_tensor.to(DEVICE)

        outputs = self.model(image_tensor)

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
            }
        }