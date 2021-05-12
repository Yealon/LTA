import json
import torch
import torch.multiprocessing
import numpy as np

from formatter.BasicFormatter import BasicFormatter
from utils.dict_utils import Dict

class Match_Formatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        # self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode

        self.src_dict = Dict(config.get("model", "src_vocab_path"))
        self.re_trg_dict = Dict(config.get("model", "re_trg_vocab_path"))
        self.re_dummy_dict = Dict(config.get("model", "dummy_vocab_path"))

        self.trg_dict = Dict(config.get("model", "trg_vocab_path"))


    def process(self, data, config, mode, *args, **params):

        src = []
        tgt = []
        raw_tgt = []
        rest_tgt = []
        # tgt_sizes = []
        meaning_id = []
        dummy_src = []
        dummy_tgt = []

        # 处理label meaning

        file_name = '/home/dkb/workspace/Code/nlp_data/dummy/meaning_dummy'
        with open(file_name, 'r', encoding="utf-8") as load_f:
            meaning_dic = json.load(load_f)
        dummy_list = self.re_dummy_dict.convertToLabels(range(self.re_dummy_dict.size()),None)

        for k, v in meaning_dic.items():
            meaning_Words = v.split()
            meaning_Words = [word for word in meaning_Words]

            # while len(meaning_Words) < self.label_max_len:
            #     meaning_Words.append('<pad>')
            # meaning_Words = meaning_Words[0:self.label_max_len]
            meaning_id += [self.src_dict.convertToIdx(meaning_Words, unkWord=True)]

        for temp in data:
            sline = temp["text"].strip()
            srcWords = sline.split()
            srcWords = [word for word in srcWords]
            # pad
            # while len(srcWords) < self.max_len:
            #     srcWords.append('<pad>')
            # srcWords = srcWords[0:self.max_len]

            src += [self.src_dict.convertToIdx(srcWords, unkWord=True)]

            tline = temp["label"].strip()
            tgtWords = tline.split()
            tgtWords = [word for word in tgtWords[:3]]

            raw_tgt += [tgtWords]
            temp_tgt = self.trg_dict.convertToIdx(tgtWords, unkWord=False, bosWord=False, eosWord=False)
            tgt += [temp_tgt]
            # tgt_sizes += [len(temp_tgt)]
            ############
            prepare_words = []
            for word in tgtWords:
                if word in dummy_list[0:4]:
                    prepare_words += ['dummy1']
                    dummy_src += [self.src_dict.convertToIdx(srcWords, unkWord=True)]
                    dummy_tgt += [
                        self.re_dummy_dict.convertToIdx(tgtWords, unkWord=False, bosWord=False, eosWord=False)]
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

            rest_tgt += [self.re_trg_dict.convertToIdx(prepare_words, unkWord=False, bosWord=False, eosWord=False)]

            # dummy_src += [self.src_dict.convertToIdx(srcWords, unkWord=True)]

        # 转变为tensor
        # 所有的案件描述 src
        src_len = [len(s) for s in src]
        src_pad = torch.zeros(len(src), max(src_len)).long()
        for i, s in enumerate(src):
            end = src_len[i]
            src_pad[i, :end] = s[:end]
        # src_len = torch.LongTensor(src_len)

        # dummy_tgt = []
        # 少数样本的案件描述 dummy_src
        if len(dummy_src) == 0:
            dum_src_pad = []
            dum_src_pad = torch.LongTensor(dum_src_pad)
        else:
            dum_src_len = [len(s) for s in dummy_src]
            dum_src_pad = torch.zeros(len(dummy_src), max(dum_src_len)).long()
            for i, s in enumerate(dummy_src):
                end = dum_src_len[i]
                dum_src_pad[i, :end] = s[:end]
            # dum_src_len = torch.LongTensor(dum_src_len)

        # 重新编号的，多数样本标签，含有dummy
        rest_tgt_len = [len(s) for s in rest_tgt]
        rest_tgt_pad = np.zeros((len(rest_tgt), self.re_trg_dict.size()), dtype=np.int)
        for i, s in enumerate(rest_tgt):
            end = rest_tgt_len[i]
            tar = s[:end]

            temp = np.zeros(self.re_trg_dict.size())
            indices = [label for label in tar]
            temp[indices] = 1
            rest_tgt_pad[i, :self.re_trg_dict.size()] = temp

        rest_tgt_pad = torch.FloatTensor(rest_tgt_pad)

        # 重新编号的，少数样本标签。
        if len(dummy_tgt) == 0:
            dummy_tgt_pad = []
        else:
            dummy_tgt_len = [len(s) for s in dummy_tgt]
            dummy_tgt_pad = np.zeros((len(dummy_tgt), self.re_dummy_dict.size()), dtype=np.int)
            for i, s in enumerate(dummy_tgt):
                end = dummy_tgt_len[i]
                tar = s[:end]

                temp = np.zeros(self.re_dummy_dict.size())
                indices = [label for label in tar]
                temp[indices] = 1
                dummy_tgt_pad[i, :self.re_dummy_dict.size()] = temp

        dummy_tgt_pad = torch.FloatTensor(dummy_tgt_pad)


        # tgt_len = [len(s) for s in tgt]
        # tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
        # for i, s in enumerate(tgt):
        #     end = tgt_len[i]
        #     tgt_pad[i, :end] = s[:end]
        # tgt_len = torch.LongTensor(tgt_len)

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

        meaning_len = [len(s) for s in meaning_id]
        meaning_pad = torch.zeros(len(meaning_id), max(meaning_len)).long()
        for i, s in enumerate(meaning_id):
            end = meaning_len[i]
            meaning_pad[i, :end] = s[:end]
        # meaning_len = torch.LongTensor(meaning_len)

        # # one hot label meaning
        # meaning_len = [len(s) for s in meaning_id]
        # meaning_pad = np.zeros((len(meaning_id), self.label_vocab.size()), dtype=np.int)
        # # meaning_pad = torch.zeros(len(meaning_id), max(meaning_len)).long()
        # for i, s in enumerate(meaning_id):
        #     end = meaning_len[i]
        #     aa = s[:end]
        #     temppp = np.zeros(self.label_vocab.size())
        #     indices = [label for label in aa]
        #     temppp[indices] = 1
        #     meaning_pad[i, :self.label_vocab.size()] = temppp
        # # meaning_len = torch.LongTensor(meaning_len)

        # 层次关系
        level12 = np.loadtxt('/home/dkb/workspace/Code/nlp_data/dummy/level-1-2.txt')
        level23 = np.loadtxt('/home/dkb/workspace/Code/nlp_data/dummy/level-2-3.txt')

        level12 = torch.from_numpy(level12)
        level23 = torch.from_numpy(level23)

        return {'src_pad': src_pad,
                # 'src_len': src_len,
                'dum_src_pad': dum_src_pad,
                # 'dum_src_len': dum_src_len,

                'tgt': tgt,
                'rest_tgt_pad': rest_tgt_pad,
                'dummy_tgt_pad': dummy_tgt_pad,

                'raw_tgt': raw_tgt,
                'meaning_pad': meaning_pad,

                'level12': level12,
                'level23': level23

                }
