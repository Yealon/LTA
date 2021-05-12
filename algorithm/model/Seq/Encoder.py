import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.getint("model", "emb_size"))
        self.rnn = nn.LSTM(input_size=config.getint("model", "emb_size"),
                           hidden_size=config.getint("model", "hidden_size"),
                           num_layers=config.getint("model", "num_layers"),
                           dropout=config.getfloat("model", "dropout"),
                           bidirectional=config.getboolean("model", "bidirec"))
        self.config = config


    def forward(self, input, lengths):

        embs = pack(self.embedding(input), lengths)
        outputs, (h, c) = self.rnn(embs)
        outputs = unpack(outputs)[0]
        # outputs:2 x embedding size,Batch_Size,2 x hidden size
        if not self.config.getboolean("model", "bidirec"):
            return outputs, (h, c)
        else:
            # outputs = outputs[:, :, :self.config.getint("model", "hidden_size")] \
            #           + outputs[:, :, self.config.getint("model", "hidden_size"):]
            # state = (h[::2], c[::2])

            # #
            batch_size = h.size(1)
            h = h.transpose(0, 1).contiguous().view(
                batch_size, -1, 2 * self.config.getint("model", "hidden_size"))
            c = c.transpose(0, 1).contiguous().view(
                batch_size, -1, 2 * self.config.getint("model", "hidden_size"))
            state = (h.transpose(0, 1), c.transpose(0, 1))
            # #
            return outputs, state