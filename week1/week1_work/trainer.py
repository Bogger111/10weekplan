import torch
import torch.nn as nn
import torch.optim as optim

def train_step(
        train_dateloader,
        model,
        optimizer,
        criterion,
        scaler,     
):
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        pred = model(X)

        optimizer.zero_grad()

        loss = criterion(pred, y)

    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(model.parameters())

    scaler.step(optimizer)

    scaler.update()

    return loss.item()

def evaluate(model, X, y):
    model.eval()

    pred = model(X)

    loss = nn.MSELoss(pred, y)

    return loss.item()

def trainer(
        X, y,
        model,
        optimizer,
        criterion,
        scaler,
        start_step = 0,
        full_step = 100,
        checkpoint_path = None
):
    try: 
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        start_step = checkpoint['global_step']
    except:
        pass

    for step in range(start_step, full_step):
        loss = train_step(X, y, model, optimizer, criterion, scaler)

        if (step + 1) % 20 == 0:
    
            print("=====Loss=====")
            print(f'step={step + 1}, loss = {loss}')
