import torch

theta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

optimizer = torch.optim.SGD([theta], lr=0.1)

loss = (theta - 0.5) ** 2

print(theta.item())
print(theta.grad)

loss.backward()

print(theta.item())
print(theta.grad.item())

optimizer.step()

print(theta.item())

print(theta.grad.item())

