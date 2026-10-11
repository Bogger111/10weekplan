import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

torch.manual_seed(91)

x = torch.randn(128, 10)

w = torch.randn(10, 1)
clean_y = x @ w 

y = clean_y + 0.1 * torch.randn_like(clean_y)

criterior = nn.MSELoss()

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

for step in range(100):
    optimizer.zero_grad()

    ypred = model(x)

    loss = criterior(y, ypred)

    loss.backward()

    optimizer.step()

    if step % 10 == 0:
        print(step, loss.item(), model[0].weight.grad.norm(), model[2].weight.grad.norm())
