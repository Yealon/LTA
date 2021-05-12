import copy

import torch
import torch.nn as nn

from model.Transformer.Batch import Batch, subsequent_mask
from model.Transformer.Beam import BeamHypotheses
from model.Transformer.SubLayers import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, Encoder, \
    Decoder, DecoderLayer, EncoderLayer, Generator, Embeddings
from model.loss import LabelSmoothing
from output.Seq.seq_index import seq_index
from utils.dict_utils import Dict
from torch.autograd import Variable

import utils.dict_utils as Dict_F


class TransModel(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    Base for this and many other models.
    """

    def __init__(self, config, *args, **params):
        super(TransModel, self).__init__()

        self.config = config

        src_vocab = Dict(config.get("model", "src_vocab_path")).size()
        tgt_vocab = Dict(config.get("model", "trg_vocab_path")).size()
        N = config.getint("model", "n_layers")
        d_model = config.getint("model", "model_dim")
        d_ff = config.getint("model", "ff_dim")
        h = config.getint("model", "n_head")
        dropout = config.getfloat("model", "dropout")

        c = copy.deepcopy  # 对象的深度copy/clone
        # 浅拷贝只引用了一级对象：copy.copy(a)
        # 深拷贝可以理解为完全独立的两个东西了：copy.deepcopy(a)
        attn = MultiHeadedAttention(h, d_model)  # 构造一个MultiHeadAttention对象。参数列表：（8, 512）

        ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 构造一个feed forward的对象，FFN。参数列表：（512, 2048, 0.1）

        position = PositionalEncoding(d_model, dropout)  # 位置编码

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.

        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        # Encoder对象
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
        # Decoder对象
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        # 源语言序列的编码，包括词嵌入和位置编码
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
        # 目标语言序列的编码，包括词嵌入和位置编码
        self.generator = Generator(d_model, tgt_vocab)
        # 生成器
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.criterion = LabelSmoothing(size=tgt_vocab,
                                        padding_idx=Dict_F.PAD, smoothing=0.01)  # 创建损失函数计算对象

    def encode(self, src, src_mask):
        # src ～ (batch.size, seq.length)
        # src_mask 负责对src加掩码
        return self.encoder(self.src_embed(src), src_mask)
        # 对源语言序列进行编码，得到的结果为～(batch.size, seq.length, 512)的tensor

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        # 对目标语言序列进行编码，得到的结果为～(batch.size, seq.length, 512)的tensor

    def forward(self, data):
        # def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."

        x = data['src_pad']
        tgt = data["tgt_pad"]
        tgt_len = data["tgt_len"]
        batch = Batch(x, tgt, Dict_F.PAD)
        out = self.decode(self.encode(batch.src, batch.src_mask), batch.src_mask, batch.trg, batch.trg_mask)
        # 先对源语言序列进行编码，将编码的结果作为memory传递给目标语言的编码器
        output = self.generator(out)
        loss = self.criterion(output.contiguous().view(-1, output.size(-1)),
                              batch.trg_y.contiguous().view(-1)) / batch.ntokens.item()
        # loss = loss_compute(out, batch.trg_y, batch.ntokens)
        return loss

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        memory = self.encode(src, src_mask)
        # 源语言的一个batch
        # 执行encode编码工作，得到memory
        # shape=(batch.size, src.seq.len, d_model)

        # src = (1,4), batch.size=1, seq.len=4
        # src_mask = (1,1,4) with all ones
        # start_symbol=1

        # print('memory={}, memory.shape={}'.format(memory,
        #                                           memory.shape))
        ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
        # 最初ys=[[1]], size=(1,1); 这里start_symbol=1
        # print('ys={}, ys.shape={}'.format(ys, ys.shape))
        for i in range(max_len - 1):  # max_len = 5
            out = self.decode(memory, src_mask,
                              Variable(ys),
                              Variable(subsequent_mask(ys.size(1))
                                       .type_as(src.data)))
            # memory, (1, 4, 8), 1=batch.size, 4=src.seq.len, 8=d_model
            # src_mask = (1,1,4) with all ones
            # out, (1, 1, 8), 1=batch.size, 1=seq.len, 8=d_model
            # print('out={}, out.shape={}'.format(out, out.shape))
            prob = self.generator(out[:, -1])
            # pick the right-most word
            # (1=batch.size,8) -> generator -> prob=(1,5) 5=trg.vocab.size
            # -1 for ? only look at the final (out) word's vector
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.unsqueeze(1)
            # word id of "next_word"
            ys = torch.cat([ys, next_word.type_as(src.data)],
                           dim=1)
            # ys is in shape of (1,2) now, i.e., 2 words in current seq
        return ys

    @seq_index
    def valid(self, data, mode):
        x = data['src_pad']
        batch = Batch(x, None, 0)
        pres = self.greedy_decode(batch.src, batch.src_mask, self.config.getint("model", "max_len"), Dict_F.BOS)

        if 'test' == mode:
            return {
                'mode': mode,
                'y_pre': pres.cpu().numpy().tolist(),
                'config': self.config,
                'raw_tgt': data['raw_tgt'],
                'y': data["tgt"],
            }
        else:
            return {'mode': mode,
                    'y_pre': pres.cpu().numpy().tolist(),
                    'y': data["tgt"],
                    'config': self.config,
                    }

