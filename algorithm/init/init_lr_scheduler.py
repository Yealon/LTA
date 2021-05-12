from torch.optim import lr_scheduler

def init_lr_scheduler(optimizer, config, leng, *args, **params):
    lr_scheduler_type = config.get("optim", "lr_scheduler")

    if lr_scheduler_type == "Step":
        step_size = config.getint("optim", "step_size")
        gamma = config.getfloat("optim", "lr_multiplier")
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler_type == "Transformer":
        optimizer.lr_scheduler(config, model_size = params['model'].src_embed[0].d_model,
                               factor=2,
                               warmup=4000)
        scheduler = None
    elif lr_scheduler_type == "Som":
        optimizer.lr_scheduler(config, leng = leng)
        scheduler = None

    else:
        raise NotImplementedError

    return scheduler
