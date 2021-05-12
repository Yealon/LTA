import torch

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from formatter.BasicFormatter import BasicFormatter

from utils.dict_utils import Dict
import numpy as np

class SeqFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        # self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.src_dict = Dict(config.get("model", "src_vocab_path"))
        self.trg_dict = Dict(config.get("model", "trg_vocab_path"))

        self.eval_dict = Dict(config.get("model", "eval_vocab_path"))

    def process(self, data, config, mode, *args, **params):
        src = []
        tgt = []
        raw_tgt = []
        tgt_sizes = []

        if "train" == mode:
            for temp in data:
                sline = temp["text"].strip()
                srcWords = sline.split()
                srcWords = [word for word in srcWords]

                # # if pad
                # while len(srcWords) < self.max_len:
                #     srcWords.append(PAD_WORD)
                # srcWords = srcWords[0:self.max_len]

                src += [self.src_dict.convertToIdx(srcWords, unkWord=True)]

                tline = temp["label"].strip()
                tgtWords = tline.split()
                tgtWords = [word for word in tgtWords[:3]]

                temp_tgt = self.trg_dict.convertToIdx(tgtWords, unkWord=True, bosWord=True, eosWord=True)
                tgt += [temp_tgt]
                tgt_sizes += [len(temp_tgt)]

            # tensor
            src_len = [len(s) for s in src]
            src_pad = torch.zeros(len(src), max(src_len)).long()
            for i, s in enumerate(src):
                end = src_len[i]
                src_pad[i, :end] = s[:end]
            src_len = torch.LongTensor(src_len)

            tgt_len = [len(s) for s in tgt]
            tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
            for i, s in enumerate(tgt):
                end = tgt_len[i]
                tgt_pad[i, :end] = s[:end]
            tgt_len = torch.LongTensor(tgt_len)

            return {'src_pad': src_pad, 'src_len': src_len,
                    'tgt_pad': tgt_pad, 'tgt_len': tgt_len}
        else:
            for temp in data:
                sline = temp["text"].strip()
                srcWords = sline.split()
                srcWords = [word for word in srcWords]

                # # if pad
                # while len(srcWords) < self.max_len:
                #     srcWords.append(PAD_WORD)
                # srcWords = srcWords[0:self.max_len]

                src += [self.src_dict.convertToIdx(srcWords, unkWord=True)]

                tline = temp["label"].strip()
                tgtWords = tline.split()

                tgtWords = [word for word in tgtWords[:3]]

                raw_tgt += [tgtWords]
                temp_tgt = self.eval_dict.convertTonumpyIdx(tgtWords, unkWord=True, bosWord=False, eosWord=False)
                tgt += [temp_tgt]
                tgt_sizes += [len(temp_tgt)]

            # tensor
            src_len = [len(s) for s in src]
            src_pad = torch.zeros(len(src), max(src_len)).long()
            for i, s in enumerate(src):
                end = src_len[i]
                src_pad[i, :end] = s[:end]
            src_len = torch.LongTensor(src_len)

            tgt_len = [len(s) for s in tgt]
            tgt_pad = np.zeros((len(tgt), self.eval_dict.size()),dtype= np.int)
            for i, s in enumerate(tgt):
                end = tgt_len[i]
                tar = s[:end]

                temp = np.zeros(self.eval_dict.size())
                indices = [label for label in tar if label is not None]
                temp[indices] = 1

                tgt_pad[i, :self.eval_dict.size()] = temp

            tgt = torch.FloatTensor(tgt_pad)

            return {'src_pad': src_pad, 'src_len': src_len,
                    'raw_tgt': raw_tgt, 'tgt': tgt}