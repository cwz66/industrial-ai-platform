import torch

x =torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print("Tensor x:")
print(x)

print("Shape:", x.shape)
print("Data type:", x.dtype)
print("Device:", x.device)

y = x*2
print("Result of x * 2:")
print(y)