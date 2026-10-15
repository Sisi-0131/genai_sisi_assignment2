from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.cnn_model import CIFAR10CNN


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "cifar10_cnn.pth"
)


image_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ]
)


class ImageClassifier:
    def __init__(self):
        self.model = CIFAR10CNN()

        model_parameters = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )

        self.model.load_state_dict(model_parameters)
        self.model.eval()

    def predict(self, image_bytes):
        image = Image.open(BytesIO(image_bytes))
        image = image.convert("RGB")

        image_tensor = image_transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        with torch.inference_mode():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)

        class_index = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, class_index].item()

        return {
            "class_index": class_index,
            "class_name": CLASS_NAMES[class_index],
            "confidence": round(confidence, 6),
        }