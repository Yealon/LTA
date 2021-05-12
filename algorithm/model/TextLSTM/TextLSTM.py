import torch
import torch.nn as nn

from model.TextLSTM.Att import Attn
from model.loss import BCEWithLogitsLoss
from output.Basic.threshold_onehot import threshold_onehot
from output.Basic.top_k_onehot import top_k_onehot
from utils.dict_utils import Dict


class TextLSTM(nn.Module):
    """
    BiLSTM: BiLSTM, BiGRU
    """

    # def __init__(self, vocab_size, embed_dim, hidden_size, use_gru, embed_dropout, fc_dropout,
    #              model_dropout, num_layers, class_num):
    def __init__(self, config, *args, **params):
        super(TextLSTM, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")

        self.embedding = nn.Embedding(Dict(config.get("model", "src_vocab_path")).size(),
                                      config.getint("model", "emb_size"))

        if config.getboolean("model", "use_gru"):
            self.bilstm = nn.GRU(config.getint("model", "emb_size"),
                                 self.hidden_size,
                                 config.getint("model", "num_layers"),
                                 dropout=(0 if config.getint("model", "num_layers") == 1 else config.getfloat("model",
                                                                                                              "model_dropout")),
                                 bidirectional=True,
                                 batch_first=True)
        else:
            self.bilstm = nn.LSTM(config.getint("model", "emb_size"),
                                  self.hidden_size,
                                  config.getint("model", "num_layers"),
                                  dropout=(0 if config.getint("model", "num_layers") == 1 else config.getfloat("model",
                                                                                                               "model_dropout")),
                                  bidirectional=True,
                                  batch_first=True)

        self.fc = nn.Linear(self.hidden_size, config.getint("model", "num_classes"))

        self.fc_dropout = nn.Dropout(config.getfloat("model", "dropout"))

        self.attn = Attn(self.hidden_size)
        self.config = config

    def forward(self, data):
        """
        :param x: [batch_size, max_len]
        :return logits: logits
        """
        x = data['src']
        tgt = data["tgt"]

        x = self.embedding(x)  # (batch_size, max_len, word_vec)
        # 输入的x是所有time step的输入, 输出的y实际每个time step的hidden输出
        # _是最后一个time step的hidden输出
        # 因为双向,y的shape为(batch_size, max_len, hidden_size*num_directions),
        # 其中[:,:,:hidden_size]是前向的结果,[:,:,hidden_size:]是后向的结果
        y, _ = self.bilstm(x)  # (batch_size, max_len, hidden_size*num_directions)
        y = y[:, :, :self.hidden_size] + y[:, :, self.hidden_size:]  # (batch_size, max_len, hidden_size)
        alpha = self.attn(y)  # (batch_size, 1, max_len)
        r = alpha.bmm(y).squeeze(1)  # (batch_size, hidden_size)
        h = torch.tanh(r)  # (batch_size, hidden_size)
        logits = self.fc(h)  # (batch_size, class_num)
        logits = self.fc_dropout(logits)

        loss = BCEWithLogitsLoss(logits, tgt)
        return loss

    @threshold_onehot
    def valid(self, data, mode):
        x = data['src']

        x = self.embedding(x)
        y, _ = self.bilstm(x)
        y = y[:, :, :self.hidden_size] + y[:, :, self.hidden_size:]
        alpha = self.attn(y)  # (batch_size, 1, max_len)
        r = alpha.bmm(y).squeeze(1)  # (batch_size, hidden_size)
        h = torch.tanh(r)  # (batch_size, hidden_size)
        logits = self.fc(h)  # (batch_size, class_num)
        logits = self.fc_dropout(logits)

        if 'test' == mode:
            return {
                'mode': mode,
                'y_pre': torch.sigmoid(logits),
                'config': self.config,
                'raw_tgt': data['raw_tgt'],
                'y': data["tgt"],
            }
        else:
            return {'mode': mode,
                    'y_pre': torch.sigmoid(logits),
                    'y': data["tgt"],
                    'config': self.config,
                    }
