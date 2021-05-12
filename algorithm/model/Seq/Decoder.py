import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .Attention import *
from .StackLSTM import StackedLSTM


class rnn_decoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None, score_fn=None):
        super(rnn_decoder, self).__init__()
        # Embedding
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.getint("model", "emb_size"))

        input_size = 2 * config.getint("model", "emb_size") if config.getboolean("model", "global_emb") else config.getint("model", "emb_size")

        # self.hidden_size = config.getint("model", "hidden_size")

        if not config.getboolean("model", "bidirec"):
            self.hidden_size = config.getint("model", "hidden_size"),
        else:
            self.hidden_size = 2 * config.getint("model", "hidden_size")

        # LSTM Cell
        self.rnn = StackedLSTM(input_size=input_size,
                               hidden_size=self.hidden_size,
                               num_layers=config.getint("model", "num_layers"),
                               dropout=config.getfloat("model", "dropout")
                               )
        # attention layer

        if config.get("model", "attention") == 'bahdanau':
            self.attention = bahdanau_attention(self.hidden_size, input_size)
        elif config.get("model", "attention") == 'luong':
            self.attention = luong_attention(self.hidden_size, input_size, config.getint("model", "pool_size"))
        elif config.get("model", "attention") == 'luong_gate':
            self.attention = luong_gate_attention(self.hidden_size, input_size)

        # output layer
        self.linear = nn.Linear(self.hidden_size, vocab_size)

        # dropout
        self.dropout = nn.Dropout(config.getfloat("model", "dropout"))

        # config
        self.config = config
        self.tgt_vocab_size = vocab_size
        # for global embedding
        if self.config.getboolean("model", "global_emb"):
            self.gated1 = nn.Linear(config.getint("model", "emb_size"), config.getint("model", "emb_size"))
            self.gated2 = nn.Linear(config.getint("model", "emb_size"), config.getint("model", "emb_size"))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, input, state, output=None, mask=None):
        embs = self.embedding(input)

        if self.config.getboolean("model", "global_emb"):
            if output is None:
                output = embs.new_zeros(embs.size(0), self.tgt_vocab_size)
            probs = self.softmax(output / self.config.getfloat("model", "tau"))
            emb_avg = torch.matmul(probs, self.embedding.weight)
            H = torch.sigmoid(self.gated1(embs) + self.gated2(emb_avg))
            emb_glb = H * embs + (1 - H) * emb_avg
            embs = torch.cat((embs, emb_glb), dim=-1)

        output, state = self.rnn(embs, state)

        if self.config.get("model", "attention") == 'luong_gate':
            output, attn_weights = self.attention(output)
        else:
            output, attn_weights = self.attention(output, embs)

        output = self.linear(output)

        if self.config.getboolean("model", "mask") and mask:
            mask = torch.stack(mask, dim=1).long()
            output.scatter_(dim=1, index=mask, value=-1e7)

        return output, state, attn_weights