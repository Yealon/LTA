from model.loss import *
# from model.loss import FocalLoss_MultiLabel
from output.Basic.threshold_onehot import threshold_onehot

from utils.dict_utils import Dict


class TextCNN(nn.Module):
    def __init__(self, config, *args, **params):
        super(TextCNN, self).__init__()
        self.config = config
        self.num_filters = config.getint("model", "num_filters")
        self.filter_sizes = [int(x) for x in config.get("model", "filter_sizes").split(',')]
        self.embedding_size = config.getint("model", "emb_size")
        self.vocab_size = Dict(config.get("model","src_vocab_path")).size()
        self.num_classes = config.getint("model", "num_classes")

        self.embed = nn.Embedding(self.vocab_size,  self.embedding_size)
        self.convs = nn.ModuleList([nn.Conv2d(1,  self.num_filters, (K, self.embedding_size)) for K in self.filter_sizes])
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))
        self.fc = nn.Linear(len(self.filter_sizes) * self.num_filters,  self.num_classes)

        # focal loss version
        # self.loss = FocalLoss_MultiLabel(alpha=0.25, gamma=2)
    def forward(self, data):

            x = data['src']
            tgt = data["tgt"]

            x = self.embed(x)  # (N, W, D)

            x = x.unsqueeze(1)  # (N, Ci, W, D)

            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

            x = torch.cat(x, 1)

            x = self.dropout(x)  # (N, len(Ks)*Co)

            logit = self.fc(x)  # (N, C)

            # focal loss version
            # index_tgt = data["index_tgt"]
            # losss = self.loss(logit, tgt)
            # losss.backward()
            # losss = losss.item()
            # return losss

            # sigmod version
            loss = BCEWithLogitsLoss(logit, tgt)
            return loss

    @threshold_onehot
    def valid(self, data, mode):
        x = data['src']


        x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)

        logit = self.fc(x)  # (N, C)
        # focal loss version
        # return F.log_softmax(logit)

        # sigmod version

        if 'test' == mode:
            return {
                'mode': mode,
                'y_pre': torch.sigmoid(logit),
                'config': self.config,
                'raw_tgt': data['raw_tgt'],
                'y': data["tgt"],
            }
        else:
            return {'mode': mode,
                    'y_pre': torch.sigmoid(logit),
                    'y': data["tgt"],
                    'config': self.config,
                    }
