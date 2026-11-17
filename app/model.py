import cv2
import torch
import numpy as np

from torchvision.models import resnet18, ResNet18_Weights

class ImageClassifier:

    def __init__(self):
        self.weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=self.weights)
        self.model.eval()

        self.preprocess = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    def predict(self, image: np.ndarray, top_k: int = 5):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb.copy()).permute(2, 0, 1)

        input_tensor = self.preprocess(tensor).unsqueeze(0)

        with torch.inference_mode():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        results = []

        for probability, index in zip(top_probs[0], top_indices[0]):
            results.append({
                "class": self.categories[index.item()],
                "confidence": round(probability.item(), 4)
            })

        return results