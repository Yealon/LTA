import torch
from transformers.tokenization_bert import BertTokenizer

from formatter.BasicFormatter import BasicFormatter
from utils.dict_utils import Dict
import numpy as np

class BasicBertFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.max_len = config.getint("data", "max_seq_length")
        self.trg_dict = Dict(config.get("model", "trg_vocab_path"))
        self.mode = mode

    def process(self, data, config, mode, *args, **params):
        input = []
        tgt = []
        raw_tgt = []

        for temp in data:
            if "trian" == mode:
                tline = temp["label"].strip()
                tgtWords = tline.split()
                tgtWords = [word for word in tgtWords[:3]]

                raw_tgt += [tgtWords]
                temp_tgt = self.trg_dict.convertTonumpyIdx(tgtWords, unkWord=False, bosWord=False, eosWord=False)
                tgt += [temp_tgt]

            text = temp["text"]
            token = self.tokenizer.tokenize(text)
            token = ["[CLS]"] + token

            while len(token) < self.max_len:
                token.append("[PAD]")
            token = token[0:self.max_len]
            token = self.tokenizer.convert_tokens_to_ids(token)

            input.append(token)

        input = torch.LongTensor(input)

        # for one hot label
        tgt_len = [len(s) for s in tgt]
        tgt_pad = np.zeros((len(tgt), self.trg_dict.size()), dtype=np.int)
        for i, s in enumerate(tgt):
            end = tgt_len[i]
            tar = s[:end]

            temp = np.zeros(self.trg_dict.size())
            indices = [label for label in tar]
            temp[indices] = 1

            tgt_pad[i, :self.trg_dict.size()] = temp

        tgt = torch.FloatTensor(tgt_pad)

        if "train" == mode:
            return {"src": input}
        else:
            return {'src': input, 'tgt': tgt}
