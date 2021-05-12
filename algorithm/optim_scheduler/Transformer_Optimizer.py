from optim_scheduler.Basic_Optimizer import BasicOptimizer


class Transformer_Optimizer(BasicOptimizer):
    def __init__(self, optimizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        # optimizer = Adam (Parameter Group 0
        #    amsgrad: False
        #    betas: (0.9, 0.98)
        #    eps: 1e-09
        #    lr: 0
        #    weight_decay: 0
        # )
        self._step = 0
        self.warmup = 0  # e.g., 4000 轮 热身
        self.factor = 0  # e.g., 2
        self.model_size = 0  # 512
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate`(learning rate) above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def lr_scheduler(self, config, *args, **kwargs):
        self.model_size = kwargs['model_size']
        self.factor = kwargs['factor']
        self.warmup = kwargs['warmup']
