import torch

a = torch.tensor([[1,0],
              [-1,0],
              [0, 1],
              [0, -1]
              ])

b = torch.tensor([-2, 0, -2, 0])

x = torch.tensor([0, 0])

res = torch.matmul(a, x)
print(res)

print(res.add(b))