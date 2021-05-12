import torch

import torch.multiprocessing

from formatter.BasicFormatter import BasicFormatter

torch.multiprocessing.set_sharing_strategy('file_system')

from utils.dict_utils import Dict
import numpy as np


class LSTMDummyFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.src_dict = Dict(config.get("model", "src_vocab_path"))
        self.trg_dict = Dict(config.get("model", "trg_vocab_path"))
        self.re_dummy_dict = Dict(config.get("model", "dummy_vocab_path"))

    def process(self, data, config, mode, *args, **params):

        src = []
        tgt = []
        raw_tgt = []

        dummy_list = self.re_dummy_dict.convertToLabels(range(self.re_dummy_dict.size()),None)

        for temp in data:
            sline = temp["text"].strip()
            srcWords = sline.split()
            srcWords = [word for word in srcWords]

            # pad
            while len(srcWords) < self.max_len:
                srcWords.append('<pad>')
            srcWords = srcWords[0:self.max_len]

            src += [self.src_dict.convertTonumpyIdx(srcWords, unkWord=True)]

            tline = temp["label"].strip()
            tgtWords = tline.split()
            tgtWords = [word for word in tgtWords[:3]]
            ############
            prepare_words = []
            for word in tgtWords:
                if word in dummy_list[0:4]:
                    prepare_words += ['dummy1']
                    continue
                elif word in dummy_list[4:13]:
                    prepare_words += ['dummy2']
                    continue
                elif word in dummy_list[13:]:
                    prepare_words += ['dummy3']
                    continue
                else:
                    prepare_words += [word]
            ##########

            raw_tgt += [prepare_words]
            temp_tgt = self.trg_dict.convertTonumpyIdx(prepare_words, unkWord=True, bosWord=False, eosWord=False)
            tgt += [temp_tgt]

        # 转变为tensor

        src = torch.LongTensor(src)

        tgt_len = [len(s) for s in tgt]
        tgt_pad = np.zeros((len(tgt), self.trg_dict.size()),dtype= np.int)
        for i, s in enumerate(tgt):
            end = tgt_len[i]
            tar = s[:end]

            temp = np.zeros(self.trg_dict.size())
            indices = [label for label in tar]
            temp[indices] = 1

            tgt_pad[i, :self.trg_dict.size()] = temp

        tgt = torch.FloatTensor(tgt_pad)

        return {'src': src, 'tgt': tgt, 'raw_tgt': raw_tgt}
