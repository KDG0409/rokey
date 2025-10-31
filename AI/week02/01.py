import torch

a = torch.arange(4.)
a = torch.reshape(a,(2,2))
print(a)

a = torch.tensor([[1,2,3],[4,5,6]])
b = 3
print(a*b)