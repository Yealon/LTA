import torch.nn as nn
import torch
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
import seaborn

seaborn.set_context(context="talk")


# seaborn只在最后可视化self-attention的时候用到，
# 可以先不管或者注释掉这两行

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # d_model=512, vocab=当前语言的词表大小
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        # one-hot转词嵌入，这里有一个待训练的矩阵E，大小是vocab*d_model
        self.d_model = d_model  # 512

    def forward(self, x):
        # x ~ (batch.size, sequence.length, one-hot),
        # one-hot大小=vocab，当前语言的词表大小
        return self.lut(x) * math.sqrt(self.d_model)
        # 得到的10*512词嵌入矩阵，主动乘以sqrt(512)=22.6，
        # 这里我做了一些对比，感觉这个乘以sqrt(512)没啥用… 求反驳。
        # 这里的输出的tensor大小类似于(batch.size, sequence.length, 512)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        # d_model=512,dropout=0.1,
        # max_len=5000代表事先准备好长度为5000的序列的位置编码，其实没必要，
        # 一般100或者200足够了。
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # (5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，
        # 每个位置用一个512维度向量来表示其位置编码
        position = torch.arange(0, max_len).unsqueeze(1)
        # torch.arange(0,max_len) result:[0,1,2,...maxlen]
        # (5000) -> (5000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # torch.arange(0, d_model, 2)：【0，2，4，...，d_model】
        # e^2i*(-log(10000)/dmodel)
        # 为每个pos乘，奇数是cos，偶数是sin，但是都针对2i
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数下标的位置
        pe = pe.unsqueeze(0)
        # (5000, 512) -> (1, 5000, 512) 为batch.size留出位置
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # 接受1.Embeddings的词嵌入结果x，
        # 然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
        # 例如，假设x是(30,10,512)的一个tensor，
        # 30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
        # 则该行代码的第二项是(1, min(10, 5000), 512)=(1,10,512)，
        # 在具体相加的时候，会扩展(1,10,512)为(30,10,512)，
        # 保证一个batch中的30个序列，都使用（叠加）一样的位置编码。
        return self.dropout(x)  # 增加一次dropout操作


# 注意，位置编码不会更新，是写死的，所以这个class里面没有可训练的参数。


def attention(query, key, value, mask=None, dropout=None):
    # query, key, value的形状类似于(30, 8, 10, 64), (30, 8, 11, 64),
    # (30, 8, 11, 64)
    # 30是batch.size，即当前batch中有多少一个序列；
    # 8=head.num，注意力头的个数；
    # 10=目标序列中词的个数
    # 11=源语言序列传过来的，当前序列的词的个数，
    # 64是每个词对应的向量表示。
    # 类似于，这里假定query来自target language sequence；
    # key和value都来自source language sequence.
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)  # 64=d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 先是(30,8,10,64)和(30, 8, 64, 11)相乘，
    # （注意是最后两个维度相乘）得到(30,8,10,11)，
    # 代表10个目标语言序列中每个词和11个源语言序列的分别的“亲密度”。
    # 然后除以sqrt(d_k)=8，防止过大的亲密度。
    # 这里的scores的shape是(30, 8, 10, 11)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 使用mask，对已经计算好的scores，按照mask矩阵，填-1e9，
    # 然后在下一步计算softmax的时候，被设置成-1e9的数对应的值~0,被忽视
    p_attn = F.softmax(scores, dim=-1)
    # 对scores的最后一个维度执行softmax，得到的还是一个tensor,
    # (30, 8, 10, 11)
    if dropout is not None:
        p_attn = dropout(p_attn)  # 执行一次dropout
    return torch.matmul(p_attn, value), p_attn
    # 返回的第一项，是(30,8,10, 11)乘以（最后两个维度相乘）value=(30,8,11,64)，得到的tensor是(30,8,10,64)，
    # 和query的最初的形状一样。另外，返回p_attn，形状为(30,8,10,11).
    # 这里返回p_attn主要是用来可视化显示多头注意力机制。


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # h=8, d_model=512
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # We assume d_v always equals d_k 512%8=0
        self.d_k = d_model // h  # d_k=512//8=64
        self.h = h  # 8
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 定义四个Linear networks, 每个的大小是(512, 512)的，
        # 每个Linear network里面有两类可训练参数，Weights，其大小为512*512，以及biases，其大小为512=d_model。

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 输入query的形状～(30, 10, 512)，
        # key.size() ~ (30, 11, 512),
        # 以及value.size() ~ (30, 11, 512)

        if mask is not None:  # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # mask下回细细分解。
        nbatches = query.size(0)  # e.g., nbatches=30
        # 1) Do all the linear projections in batch from
        # d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k)
                                 .transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 这里是前三个Linear Networks的具体应用，
        # 例如query=(30,10, 512) -> Linear network -> (30, 10, 512)
        #                   -> view -> (30,10, 8, 64) -> transpose(1,2) -> (30, 8, 10, 64)
        # ，其他的key和value也是类似地，
        # 从(30, 11, 512) -> (30, 8, 11, 64)。
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # 调用上面定义好的attention函数，输出的x形状为(30, 8, 10, 64)；
        # attn的形状为(30, 8, 10=target.seq.len, 11=src.seq.len)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # x ~ (30, 8, 10, 64) -> transpose(1,2) ->
        # (30, 10, 8, 64) -> contiguous() and view ->
        # (30, 10, 8*64) = (30, 10, 512)
        return self.linears[-1](x)
        # 执行第四个Linear network，把(30, 10, 512)经过一次linear network，
        # 得到(30, 10, 512).


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        # features=d_model=512, eps=epsilon 用于分母的非0化平滑
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        # a_2 是一个可训练参数向量，(512)
        self.b_2 = nn.Parameter(torch.zeros(features))
        # b_2 也是一个可训练参数向量, (512)
        self.eps = eps

    def forward(self, x):
        # x 的形状为(batch.size, sequence.len, 512)
        mean = x.mean(-1, keepdim=True)
        # 对x的最后一个维度，取平均值，得到tensor (batch.size, seq.len)
        std = x.std(-1, keepdim=True)
        # 对x的最后一个维度，取标准方差，得(batch.size, seq.len)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # 本质上类似于（x-mean)/std，不过这里加入了两个可训练向量
        # a_2 and b_2，以及分母上增加一个极小值epsilon，用来防止std为0的时候的除法溢出


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    残差Add以及Norm
    """

    def __init__(self, size, dropout):
        # size=d_model=512; dropout=0.1
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)  # (512)，用来定义a_2和b_2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the "
        "same size."
        # x is alike (batch.size, sequence.len, 512)
        # sublayer是一个具体的MultiHeadAttention或者PositionwiseFeedForward对象
        return x + self.dropout(sublayer(self.norm(x)))
        # x (30, 10, 512) -> norm (LayerNorm) -> (30, 10, 512)
        #                -> sublayer (MultiHeadAttention or PositionwiseFeedForward)
        #                       -> (30, 10, 512) -> dropout -> (30, 10, 512)
        # 然后输入的x（没有走sublayer) + 上面的结果，即实现了残差相加的功能


class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    FFN = max(0,xW1+b1)W2+b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model = 512
        # d_ff = 2048 = 512*4
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        # 构建第一个全连接层，(512, 2048)，其中有两种可训练参数：
        # weights矩阵，(512, 2048)，以及
        # biases偏移向量, (2048)
        self.w_2 = nn.Linear(d_ff, d_model)
        # 构建第二个全连接层, (2048, 512)，两种可训练参数：
        # weights矩阵，(2048, 512)，以及
        # biases偏移向量, (512)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape = (batch.size, sequence.len, 512)
        # 例如, (30, 10, 512)
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        # x (30, 10, 512) -> self.w_1 -> (30, 10, 2048)
        #       -> relu -> (30, 10, 2048)
        #           -> dropout -> (30, 10, 2048)
        #                -> self.w_2 -> (30, 10, 512)是输出的shape


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        # size=d_model=512
        # self_attn = MultiHeadAttention对象, first sublayer
        # feed_forward = PositionwiseFeedForward对象，second sublayer
        # dropout = 0.1 (e.g.)
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 使用深度克隆方法，完整地复制出来两个SublayerConnection
        self.size = size  # 512

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        # x shape = (30, 10, 512)
        # mask 是(batch.size, 10,10)的矩阵，类似于当前一个词w，有哪些词是w可见的
        # 源语言的序列的话，所有其他词都可见，除了"<blank>"这样的填充；
        # 目标语言的序列的话，所有w的左边的词，都可见。
        x = self.sublayer[0](x,
                             lambda x: self.self_attn(x, x, x, mask))
        # 这里的lambda函数类似：return self.self_attn(x, x, x, mask)
        # x (30, 10, 512) -> self_attn (MultiHeadAttention)
        # shape is same (30, 10, 512) -> SublayerConnection
        # -> (30, 10, 512)
        return self.sublayer[1](x, self.feed_forward)  # 第一个的半个block，multi-head之后feed forward
        # x 和feed_forward对象一起，给第二个SublayerConnection


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        # layer = one EncoderLayer object, N=6
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        # 深copy，N=6，
        self.norm = LayerNorm(layer.size)
        # 定义一个LayerNorm，layer.size=d_model=512
        # 其中有两个可训练参数a_2和b_2

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        # x is alike (30, 10, 512)
        # (batch.size, sequence.len, d_model)
        # mask是类似于(batch.size, 10, 10)的矩阵
        for layer in self.layers:
            x = layer(x, mask)
            # 进行六次EncoderLayer操作
        return self.norm(x)
        # 最后做一次LayerNorm，最后的输出也是(30, 10, 512) shape候的除法溢出


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        # size = d_model=512,
        # self_attn = first MultiHeadAttention object，目标语言序列的
        # src_attn = second MultiHeadAttention object, 目标语言序列和源语言序列之间的
        # feed_forward 一个全连接层
        # dropout = 0.1
        super(DecoderLayer, self).__init__()
        self.size = size  # 512
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        # 需要三个SublayerConnection, 分别在
        # self.self_attn, self.src_attn, 和self.feed_forward
        # 的后边

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory  # (batch.size, sequence.len, 512)
        # 来自源语言序列的Encoder之后的输出，作为memory，供目标语言的序列检索匹配：（类似于alignment in SMT)
        x = self.sublayer[0](x,
                             lambda x: self.self_attn(x, x, x, tgt_mask))
        # 通过lambda（匿名函数），来实现目标序列的自注意力编码，结果再过一个SublayerConnection
        x = self.sublayer[1](x,
                             lambda x: self.src_attn(x, m, m, src_mask))
        # 通过第二个lambda（匿名函数），来实现目标序列和源序列的注意力计算，结果继续过SublayerConnection
        return self.sublayer[2](x, self.feed_forward)
        # 走feed_forward的全连接层，然后结果扔给一个SublayerConnection


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        # layer = DecoderLayer object
        # N = 6
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        # 深度copy六次DecoderLayer
        self.norm = LayerNorm(layer.size)
        # 初始化一个LayerNorm

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            # 执行六次DecoderLayer
        return self.norm(x)
        # 执行一次LayerNorm


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    输出层：linear + softmax
    """

    def __init__(self, d_model, vocab):
        # d_model=512
        # vocab = 目标语言词表大小
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        # 定义一个全连接层，可训练参数个数是(512 * trg_vocab_size) +
        # trg_vocab_size

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
        # x 类似于 (batch.size, sequence.length, 512)
        #               -> proj 全连接层 (30, 10, trg_vocab_size) = logits
        # 对最后一个维度执行log_soft_max，得到(30, 10, trg_vocab_size)
