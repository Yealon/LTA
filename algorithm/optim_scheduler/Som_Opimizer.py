from optim_scheduler.Basic_Optimizer import BasicOptimizer


class Som_Optimizer(BasicOptimizer):
    def __init__(self, enc_optimizer, dec_optimizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enc_optimizer = enc_optimizer
        self.enc_scheduler = None

        self.dec_optimizer = dec_optimizer
        self.dec_scheduler = None

    def step(self):
        "Update parameters and rate"
        self.enc_optimizer.step()
        self.enc_scheduler.step()
        self.dec_optimizer.step()
        self.dec_scheduler.step()

    # def get_lr(self):
    #     """Return the current learning rate."""
    #     k = (
    #         "default"
    #         if "default" in self.optimizers
    #         else next(iter(self.optimizers.keys()))
    #     )
    #     return self.optimizers[k].param_groups[0]["lr"]

    def state_dict(self):
        """Return the state dict."""
        return {'enc_optimizer': self.enc_optimizer.state_dict(),
                'dec_optimizer': self.dec_optimizer.state_dict()
                }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.enc_optimizer.load_state_dict(state_dict['enc_optimizer'])
        self.dec_optimizer.load_state_dict(state_dict['dec_optimizer'])

    def lr_scheduler(self, config, *args, **kwargs):
        leng = kwargs['leng']

        self.enc_scheduler = get_linear_schedule_with_warmup(self.enc_optimizer,
                                                        int(leng * config.getint("train", "epoch") * config.getfloat(
                                                            "optim", "warmup_rate")),
                                                        int(leng * config.getint("train", "epoch")))
        self.dec_scheduler = get_linear_schedule_with_warmup(self.dec_optimizer,
                                                        int(leng * config.getint("train", "epoch") * config.getfloat(
                                                            "optim", "warmup_rate")),
                                                        int(leng * config.getint("train", "epoch")))

from transformers import AdamW, get_linear_schedule_with_warmup


def get_som_opt(model, config):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    enc_param_optimizer = list(model.encoder.named_parameters())
    enc_optimizer_grouped_parameters = [
        {'params': [p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=config.getfloat("optim", "enc_lr"))

    dec_param_optimizer = list(model.decoder.parameters())
    dec_optimizer = AdamW(dec_param_optimizer, lr=config.getfloat("optim", "dec_lr"))

    return Som_Optimizer(enc_optimizer, dec_optimizer)