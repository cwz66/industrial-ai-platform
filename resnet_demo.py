import cv2
import torch

from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

model.eval()

preprocess = weights.transforms()

image = cv2.imread("images/test.jpg")
if image is None:
    raise ValueError("Cannot read images/test.jpg")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

tensor = torch.from_numpy(image_rgb.copy()).permute(2, 0, 1)

input_tensor = preprocess(tensor).unsqueeze(0)

print("Input shape:", input_tensor.shape)

with torch.inference_mode():
    output = model(input_tensor)

print("Output shape:", output.shape)

probabilities = torch.softmax(output, dim=1)

predicted_index = probabilities.argmax(dim=1).item()

categories = weights.meta["categories"]
predicted_class = categories[predicted_index]

confidence = probabilities[0, predicted_index].item()

top_probs, top_indices = torch.topk(
    probabilities,
    k=5,
    dim=1
)

print("\nTop-5 predictions:")

for i in range(5):
    class_index = top_indices[0, i].item()
    probability = top_probs[0, i].item()

    class_name = categories[class_index]

    print(
        f"{i + 1}. {class_name}: "
        f"{probability * 100:.2f}%"
    )