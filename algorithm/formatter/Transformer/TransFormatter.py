import torch

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from formatter.BasicFormatter import BasicFormatter

from utils.dict_utils import Dict
import numpy as np


class TransFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        # self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.max_len = config.getint("data", "max_seq_length")
        self.src_dict = Dict(config.get("model", "src_vocab_path"))
        self.trg_dict = Dict(config.get("model", "trg_vocab_path"))

        self.eval_dict = Dict(config.get("model", "eval_vocab_path"))

    def process(self, data, config, mode, *args, **params):
        src = []
        tgt = []
        raw_tgt = []
        tgt_sizes = []

        if "train" == mode:
            # 将src，进target行转化为id，添加bos与eos
            for temp in data:
                sline = temp["text"].strip()
                srcWords = sline.split()
                srcWords = [word for word in srcWords]

                # if pad
                while len(srcWords) < self.max_len:
                    srcWords.append('<pad>')
                srcWords = srcWords[0:self.max_len]

                src += [self.src_dict.convertTonumpyIdx(srcWords, unkWord=True)]

                tline = temp["label"].strip()
                tgtWords = tline.split()
                tgtWords = [word for word in tgtWords[:3]]

                temp_tgt = self.trg_dict.convertToIdx(tgtWords, unkWord=True, bosWord=True, eosWord=True)
                tgt += [temp_tgt]
                tgt_sizes += [len(temp_tgt)]

            # 转变为tensor，其中target需要进行pad，保证统一长度。
            # 例如：[bos,1,2,3,eos] 与 [bos,1,2,eos,pad]
            # 通过loss将pad的影响消去

            tgt_len = [len(s) for s in tgt]
            tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
            for i, s in enumerate(tgt):
                end = tgt_len[i]
                tgt_pad[i, :end] = s[:end]
            tgt_len = torch.LongTensor(tgt_len)

            src = torch.LongTensor(src)
            return {'src_pad': src,
                    'tgt_pad': tgt_pad, 'tgt_len': tgt_len}
        else:
            # 验证集/测试集，保存每个输入的target原始文本与去掉bos,eos的id，方便计算指标
            # 验证/测试过程中不需target，只输入bos进行一系列decode，因此不需要对target进行处理
            for temp in data:
                sline = temp["text"].strip()
                srcWords = sline.split()
                srcWords = [word for word in srcWords]

                # if pad
                while len(srcWords) < self.max_len:
                    srcWords.append('<pad>')
                srcWords = srcWords[0:self.max_len]

                src += [self.src_dict.convertTonumpyIdx(srcWords, unkWord=True)]

                tline = temp["label"].strip()
                tgtWords = tline.split()

                tgtWords = [word for word in tgtWords[:3]]

                raw_tgt += [tgtWords]
                temp_tgt = self.eval_dict.convertTonumpyIdx(tgtWords, unkWord=True, bosWord=False, eosWord=False)
                tgt += [temp_tgt]
                tgt_sizes += [len(temp_tgt)]

            # 转化为tensor,为方便计算指标，target直接转化为独热编码

            tgt_len = [len(s) for s in tgt]
            tgt_pad = np.zeros((len(tgt), self.eval_dict.size()), dtype=np.int)
            for i, s in enumerate(tgt):
                end = tgt_len[i]
                tar = s[:end]
                temp = np.zeros(self.eval_dict.size())
                indices = [label for label in tar]
                temp[indices] = 1

                tgt_pad[i, :self.eval_dict.size()] = temp

            tgt = torch.FloatTensor(tgt_pad)

            src = torch.LongTensor(src)
            return {'src_pad': src,
                    'raw_tgt': raw_tgt, 'tgt': tgt}

