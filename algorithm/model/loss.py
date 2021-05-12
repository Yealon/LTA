import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import utils.dict_utils as Dict_F


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss


class LabelSmoothing(nn.Module):
    """Implement label smoothing.
        LabelSmoothingKLLoss
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx  # '<blank>' 的id
        self.confidence = 1.0 - smoothing  # 自留的概率值、得分 e.g. 0.6
        self.smoothing = smoothing  # 均分出去的概率值，得分 e.g. 0.4
        self.size = size  # target vocab size 目标语言词表大小
        self.true_dist = None

    def forward(self, x, target):
        "in real-world case: 真实情况下"
        # x的shape为(batch.size * seq.len, target.vocab.size)
        # y的shape是(batch.size * seq.len)

        # x=logits，(seq.len, target.vocab.size)
        # 每一行，代表一个位置的词
        # 类似于：假设seq.len=3, target.vocab.size=5
        # x中保存的是log(prob)
        # x = tensor([[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233]])

        # target 类似于：
        # target = tensor([2, 1, 0])，torch.size=(3)

        assert x.size(1) == self.size  # 目标语言词表大小
        true_dist = x.data.clone()
        # true_dist = tensor([[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233]])

        true_dist.fill_(self.smoothing / (self.size - 2))
        # true_dist = tensor([[0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        # [0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        # [0.1333, 0.1333, 0.1333, 0.1333, 0.1333]])

        # 注意，这里分母target.vocab.size-2是因为
        # (1) 最优值 0.6要占一个位置；
        # (2) 填充词 <blank> 要被排除在外
        # 所以被激活的目标语言词表大小就是self.size-2

        true_dist.scatter_(1, target.data.unsqueeze(1),
                           self.confidence)
        # target.data.unsqueeze(1) ->
        # tensor([[2],
        # [1],
        # [0]]); shape=torch.Size([3, 1])
        # self.confidence = 0.6

        # 根据target.data的指示，按照列优先(1)的原则，把0.6这个值
        # 填入true_dist: 因为target.data是2,1,0的内容，
        # 所以，0.6填入第0行的第2列（列号，行号都是0开始）
        # 0.6填入第1行的第1列
        # 0.6填入第2行的第0列：
        # true_dist = tensor([[0.1333, 0.1333, 0.6000, 0.1333, 0.1333],
        # [0.1333, 0.6000, 0.1333, 0.1333, 0.1333],
        # [0.6000, 0.1333, 0.1333, 0.1333, 0.1333]])

        true_dist[:, self.padding_idx] = 0
        # true_dist = tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
        # [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
        # [0.0000, 0.1333, 0.1333, 0.1333, 0.1333]])
        # 设置true_dist这个tensor的第一列的值全为0
        # 因为这个是填充词'<blank>'所在的id位置，不应该计入
        # 目标词表。需要注意的是，true_dist的每一列，代表目标语言词表
        # 中的一个词的id

        mask = (target.data == self.padding_idx).nonzero()
        # mask = tensor([[2]]), 也就是说，最后一个词 2,1,0中的0，
        # 因为是'<blank>'的id，所以通过上面的一步，把他们找出来

        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            # 当target reference序列中有0这个'<blank>'的时候，则需要把
            # 这一行的值都清空。
            # 在一个batch里面的时候，可能两个序列长度不一，所以短的序列需要
            # pad '<blank>'来填充，所以会出现类似于(2,1,0)这样的情况
            # true_dist = tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
            # [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
            # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
        self.true_dist = true_dist
        return self.criterion(x,
                              Variable(true_dist, requires_grad=False))
        # 这一步就是调用KL loss来计算
        # x = tensor([[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233]])

        # true_dist=tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
        # [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])

        # 之间的loss了。

        # LabelSmoothing，一方面对label进行平滑，如果Model对于一个结果非常确信，则loss反而惩罚它（貌似缺少了多样性）；另外一方面则是对loss进行计算的。


# 针对二分类任务的 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
        pred = nn.Sigmoid()(pred)

        # 展开 pred 和 target,此时 pred.size = target.size = (BatchSize,1)
        pred = pred.view(-1, 1)
        target = target.view(-1, 1)

        # 此处将预测样本为正负的概率都计算出来，此时 pred.size = (BatchSize,2)
        pred = torch.cat((1 - pred, pred), dim=1)

        # 根据 target 生成 mask，即根据 ground truth 选择所需概率
        # 用大白话讲就是：
        # 当标签为 1 时，我们就将模型预测该样本为正类的概率代入公式中进行计算
        # 当标签为 0 时，我们就将模型预测该样本为负类的概率代入公式中进行计算
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()
        # 这里的 scatter_ 操作不常用，其函数原型为:
        # scatter_(dim,index,src)->Tensor
        # Writes all values from the tensor src into self at the indices specified in the index tensor.
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 将所需概率值挑选出来
        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        # 计算概率的 log 值
        log_p = probs.log()

        # 根据论文中所述，对 alpha　进行设置（该参数用于调整正负样本数量不均衡带来的问题）
        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        # 根据 Focal Loss 的公式计算 Loss
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        # Loss Function的常规操作，mean 与 sum 的区别不大，相当于学习率设置不一样而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


# 针对 Multi-Label 任务的 Focal Loss
class FocalLoss_MultiLabel(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss_MultiLabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        criterion = FocalLoss(self.alpha, self.gamma, self.size_average)
        loss = torch.zeros(1, target.shape[1]).cuda()

        # 对每个 Label 计算一次 Focal Loss
        for label in range(target.shape[1]):
            batch_loss = criterion(pred[:, label], target[:, label])
            loss[0, label] = batch_loss.mean()

        # Loss Function的常规操作
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss


class FocalLoss_Level(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss_Level, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.loss1 = FocalLoss_MultiLabel(alpha=0.5, gamma=2)
        self.loss2 = FocalLoss_MultiLabel(alpha=0.5, gamma=2)
        self.loss3 = FocalLoss_MultiLabel(alpha=0.5, gamma=2)

    def forward(self, outputs1, outputs2, outputs3, labels):
        loss1 = self.loss1(outputs1, labels[:, 0:8])
        loss2 = self.loss2(outputs2, labels[:, 8:22])
        loss3 = self.loss3(outputs3, labels[:, 22:])

        loss = loss1 + 1.5 * loss2 + 2 * loss3

        # loss.backward()
        # loss = loss.item()

        return loss


def BCEWithLogitsLoss(outputs, labels):
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(outputs, labels)
    # loss.backward()
    # loss = loss.item()

    return loss


def BCEWithLogitsLoss_level(outputs1, outputs2, outputs3, labels):
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.BCEWithLogitsLoss()
    criterion3 = nn.BCEWithLogitsLoss()

    loss1 = criterion1(outputs1, labels[:, 0:8])
    loss2 = criterion2(outputs2, labels[:, 8:22])
    loss3 = criterion3(outputs3, labels[:, 22:])

    loss = loss1 + loss2 + loss3

    # loss.backward()
    # loss = loss.item()

    return loss


def Seq2Seq_loss(outputs, targets, sim_score=0.001):
    outputs = outputs.view(-1, outputs.size(2))
    targets = targets.contiguous().view(-1)

    loss = nn.CrossEntropyLoss(ignore_index=Dict_F.PAD, reduction='none')(outputs, targets) + sim_score

    num_total = targets.ne(Dict_F.PAD).sum().item()
    # reduction = none, 输出样本数相等的向量【loss1,loss2...】，根据target中除PAD之外的数量自行求平均值
    loss = torch.sum(loss) / num_total
    # loss.backward()

    return loss
