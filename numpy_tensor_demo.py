import numpy as np
import torch

array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
print("NumPy array:")
print(array)

tensor = torch.from_numpy(array)
print("\nPyTorch Tensor:")
print(tensor)

print("Array shape:", array.shape)
print("Tensor shape:", tensor.shape)
print("Tensor dtype:", tensor.dtype)

array[0, 0] = 100.0

print("\nAfter modifying NumPy array:")
print(array)
print(tensor)