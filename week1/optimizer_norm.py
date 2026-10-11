import copy
import torch
import torch.nn as nn
from train_toy import make_data, build_model, train_step, val

X_train, y_train, X_val, y_val = make_data(42)

criterion = nn.MSELoss()

base_model = build_model('id')

inial_linear0 = copy.deepcopy(base_model[0].state_dict())
inial_linear3 = copy.deepcopy(base_model[3].state_dict())


def run_optimizer(name, norm = "id"):
    model = build_model(norm)
    model[0].load_state_dict(inial_linear0)
    model[3].load_state_dict(inial_linear3)

    if name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr = 0.01
        )

    elif name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr = 0.01
        )

    batch_size = 3

    first_para_change = None

    for step in range(100):
        start = (step * batch_size) % len(X_train)
        end = start + batch_size

        X_batch = X_train[start: end]
        y_bacth = y_train[start: end]

        if step == 0:
            before = model[0].weight.detach().clone()

        loss, grad_norm = train_step(
            model,
            optimizer,
            criterion,
            X_batch,
            y_bacth
        )

        if step == 0:
            after = model[0].weight.detach().clone()
            first_para_change = (after - before).norm().item()

    train_mse = val(
        model, criterion, X_train, y_train
    )

    val_mse = val(
        model, criterion, X_val, y_val
    )

    state = optimizer.state_dict()['state']


    print(
        name,
        "train =", train_mse,
        "val =", val_mse,
        "first update =", first_para_change,
        "number_state =", len(optimizer.state_dict()["state"]),
        "="*100
    )
    if state:
        print("state", next(iter(state.values())).keys())



run_optimizer('sgd', norm='id')
run_optimizer('sgd', norm='bn')
run_optimizer('sgd', norm='ln')
run_optimizer('adam', norm='id')
run_optimizer('adam', norm='bn')
run_optimizer("adam", norm='ln')


