import torch.optim as optim

from optim_scheduler.Som_Opimizer import get_som_opt
from optim_scheduler.Transformer_Optimizer import Transformer_Optimizer


def init_optimizer(model, config, *args, **params):
    optimizer_type = config.get("optim", "optimizer")
    learning_rate = config.getfloat("optim", "learning_rate")
    if optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=config.getfloat("optim", "weight_decay"))
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              weight_decay=config.getfloat("optim", "weight_decay"))
    elif optimizer_type == "transformer":
        optimizer = Transformer_Optimizer(optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    elif optimizer_type == "Som":
        optimizer = get_som_opt(model, config)

    else:
        raise NotImplementedError

    return optimizer
