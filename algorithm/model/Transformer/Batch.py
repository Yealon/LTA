from torch.autograd import Variable
import numpy as np
import torch

def subsequent_mask(size):
    """Mask out subsequent positions."""
    # e.g., size=10
    attn_shape = (1, size, size)  # (1, 10, 10)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # triu: 负责生成一个三角矩阵，k-th对角线以下都是设置为0，上三角中元素为1.
    # 1号对角线的形式：对角线全0，以上为1，以下为0
    # 0号对角线：对角线全1，以上为1，以下为0。。。以此类推

    return torch.from_numpy(subsequent_mask) == 0
    # 反转上面的triu得到的上三角矩阵，修改为下三角矩阵。


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, trg=None, pad=0):
        # src: 源语言序列，(batch.size, src.seq.len)
        # 二维tensor，第一维度是batch.size；第二个维度是源语言句子的长度
        # 例如：[ [2,1,3,4], [2,3,1,4] ]这样的二行四列的，
        # 1-4代表每个单词word的id

        # trg: 目标语言序列，默认为空，其shape和src类似
        # (batch.size, trg.seq.len)，
        # 二维tensor，第一维度是batch.size；第二个维度是目标语言句子的长度
        # 例如trg=[ [2,1,3,4], [2,3,1,4] ] for a "copy network"
        # (输出序列和输入序列完全相同）

        # pad: 源语言和目标语言统一使用的 位置填充符号，'<blank>'
        # 所对应的id，这里默认为0
        # 例如，如果一个source sequence，长度不到4，则在右边补0
        # [1,2] -> [1,2,0,0]

        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        # src = (batch.size, seq.len) -> != pad ->
        # (batch.size, seq.len) -> usnqueeze ->
        # (batch.size, 1, seq.len) 相当于在倒数第二个维度扩展
        # e.g., src=[ [2,1,3,4], [2,3,1,4] ]对应的是
        # src_mask=[ [[1,1,1,1], [1,1,1,1]] ]
        if trg is not None:
            self.trg = trg[:, :-1]  # 重要
            # trg 相当于目标序列的前N-1个单词的序列（去掉了最后一个词）
            self.trg_y = trg[:, 1:]
            # trg_y 相当于目标序列的后N-1个单词的序列(去掉了第一个词）
            # 目的是(src + trg) 来预测出来(trg_y)，
            # 这个在illustrated transformer中详细图示过。
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    #  静态方法无需实例化，C.f(); cobj.f()：也可以实例化后调用
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        # 这里的tgt类似于：
        # [ [2,1,3], [2,3,1] ] （最初的输入目标序列，分别去掉了最后一个词
        # pad=0, '<blank>'的id编号
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # 得到的tgt_mask类似于
        # tgt_mask = tensor([[[1, 1, 1]],[[1, 1, 1]]], dtype=torch.uint8)
        # shape=(2,1,3)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        # 先看subsequent_mask, 其输入的是tgt.size(-1)=3
        # 这个函数的输出为= tensor([[[1, 0, 0],
        #                          [1, 1, 0],
        #                          [1, 1, 1]]], dtype=torch.uint8) 下三角矩阵
        # type_as 把这个tensor转成tgt_mask.data的type(也是torch.uint8)

        # 这样的话，&的两边的tensor形状分别是(2,1,3), (1,3,3);
        # tgt_mask = tensor([[[1, 1, 1]],[[1, 1, 1]]], dtype=torch.uint8)
        # and
        # tensor([[[1, 0, 0], [1, 1, 0], [1, 1, 1]]], dtype=torch.uint8)

        # 得到的tensor形状：(2,3,3)
        # tgt_mask.data = tensor([[[1, 0, 0],
        #                          [1, 1, 0],
        #                          [1, 1, 1]],

        # 关于&
        # &算子左边，当tgt_mask,shape=(3,1,4)是：
        # [1 1 1 1],
        # [1 1 1 0],
        # [1 1 1 1]
        # 类似于第二个序列的最后一个词是'<blank>' 。

        # &算子右边，的Variable()内部，shape=(1,4,4)，
        # 是：
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        # 的时候。

        # 这两者的&的结果是 (3, 4, 4)：
        # [1 1 1 1] &
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        # =
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1

        # [1 1 1 0] &
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        # =
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 0 # attention here， 因为第二个序列只有3个词

        # [1 1 1 1] &
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1
        # =
        # 1 0 0 0
        # 1 1 0 0
        # 1 1 1 0
        # 1 1 1 1

        # 我的理解：与乘法一样，只不过逐个点乘变为了逐个&
        # 关于&

        return tgt_mask
