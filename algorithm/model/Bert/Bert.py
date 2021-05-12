import torch.nn as nn
import torch
from transformers import BertModel

from model.loss import BCEWithLogitsLoss


class BasicBert(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BasicBert, self).__init__()

        self.output_dim = config.getint("model", "num_classes")
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        self.fc = nn.Linear(768, self.output_dim)

        # self.criterion = nn.CrossEntropyLoss()

    def init_multi_gpu(self, device, config, *args, **params):
        self.bert = nn.DataParallel(self.bert, device_ids=device)

    def forward(self, data, config, gpu_list, acc_result, mode):
        x = data['input']

        _, y = self.bert(x)
        y = y.view(y.size()[0], -1)
        y = self.fc(y)
        y = y.view(y.size()[0], -1)

        if "train" == mode:
            tgt = data["tgt"]
            loss = BCEWithLogitsLoss(y, tgt)
            return loss
            # label = data["label"]
            # loss = self.criterion(y, label)
        else:
            return torch.sigmoid(y)
