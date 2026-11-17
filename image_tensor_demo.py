import cv2
import torch
import numpy as np

image = np.full((100, 200, 3), 128, dtype=np.uint8)
print("Original Image Shape:", image.shape)
print("Original Image dType:", image.dtype)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

tensor = torch.from_numpy(image_rgb.copy()).float() / 255.0

tensor = tensor.permute(2, 0, 1)

tensor = tensor.unsqueeze(0)

print("Final Tensor shape:", tensor.shape)
print("Final Tensor dtype:", tensor.dtype)
print("Minimum value:", tensor.min().item())
print("Maximum value:", tensor.max().item())