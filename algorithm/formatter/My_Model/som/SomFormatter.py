import torch

import torch.multiprocessing
from transformers import BertTokenizer

torch.multiprocessing.set_sharing_strategy('file_system')

from formatter.BasicFormatter import BasicFormatter

from utils.dict_utils import Dict
import numpy as np

OP_SET = {
    '0': {'合同、无因管理、不当得利纠纷': 0, '侵权责任纠纷1': 1, '婚姻家庭、继承纠纷': 2, '人格权纠纷1': 3, '物权纠纷': 4, '与公司、证券、保险、票据等有关的民事纠纷': 5, '劳动争议、人事争议': 6, '知识产权与竞争纠纷': 7},
    # '3-1': {'update': 0, 'carryover': 1, 'dontcare': 2},
    # '3-2': {'update': 0, 'carryover': 1, 'delete': 2},
    '4': {'合同纠纷': 0, '侵权责任纠纷': 1, '婚姻家庭纠纷': 2, '人格权纠纷': 3, '物权保护纠纷': 4, '期货交易纠纷': 5, '保险纠纷': 6, '与企业有关的纠纷': 7,
          '劳动争议': 8, '与破产有关的纠纷': 9, '与公司有关的纠纷': 10, '用益物权纠纷': 11, '不当得利纠纷2': 12, '知识产权权属、侵权纠纷': 13}
}

flatten = lambda x: [i for s in x for i in s]

class SomFormatter(BasicFormatter):
    def __init__(self, config, mode, *args, **params):
        super().__init__(config, mode, *args, **params)

        # self.max_len = config.getint("data", "max_seq_length")
        self.mode = mode
        self.max_len = config.getint("data", "max_seq_length")
        # self.tokenizer = BertTokenizer(config.get("model", "tokenizer"))
        self.tokenizer = BertTokenizer.from_pretrained(config.get("model", "bert_path"))
        self.state = []
        self.eval_dict = Dict(config.get("model", "eval_vocab_path"))

        for s in [a for a in OP_SET['4'].keys()]:
            self.state.append('[SLOT]')
            self.state.extend(self.tokenizer.tokenize(' '.join(s)))

    def process(self, data, config, mode, *args, **params):
        src = []
        segment_ids=[]
        input_masks=[]
        state_position_ids=[]
        op_ids = []
        domain_ids = []
        gen_ids = []
        # op2ids=[]

        raw_tgt = []
        tgt = []
        tgt_sizes = []


        if "train" == mode:
            # 将src，进target行转化为id，添加bos与eos

            for temp in data:
                state = self.state
                sline = temp["text"].strip()

                # if pad
                avail_src_length = self.max_len - len(state) - 3
                # while len(sline) < self.max_len:
                #     sline = sline + '<pad>'
                #
                # sline = sline[0:self.max_len]

                src_tokens = self.tokenizer.tokenize(' '.join(sline))

                if len(src_tokens) > avail_src_length:
                    # avail_length = len(src_tokens) - avail_src_length
                    src_tokens = src_tokens[:avail_src_length]

                drop_mask = [0] + [0] * len(state) + [0] + [1] * len(src_tokens) + [0]
                diag_1 = ["[CLS]"] + state + ["[SEP]"]
                diag_2 = src_tokens + ["[SEP]"]
                segment = [0] * len(diag_1) + [1] * len(diag_2)

                diag = diag_1 + diag_2
                # word dropout
                if config.getfloat("model", "word_dropout") > 0.:
                    drop_mask = np.array(drop_mask)
                    word_drop = np.random.binomial(drop_mask.astype('int64'), config.getfloat("model", "word_dropout"))
                    diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]

                input_ = diag
                segment_id = segment

                slot_position = []
                for i, t in enumerate(input_):
                    if t == '[SLOT]':
                        slot_position.append(i)

                input_mask = [1] * len(input_)
                input_id = self.tokenizer.convert_tokens_to_ids(input_)
                if len(input_mask) < self.max_len:
                    input_id = input_id + [0] * (self.max_len - len(input_mask))
                    segment_id = segment_id + [0] * (self.max_len - len(input_mask))
                    input_mask = input_mask + [0] * (self.max_len - len(input_mask))

                # append
                src += [input_id]
                segment_ids += [segment_id]
                input_masks += [input_mask]
                state_position_ids += [slot_position]

                # append
                tline = temp["label"].strip()
                tgtWords = tline.split()
                try:
                    labels = tgtWords[2]
                except IndexError as e:
                    labels = tgtWords[1]
                domain_ids += [OP_SET['0'][tgtWords[0]]]
                op_id =OP_SET['4'][tgtWords[1]]
                # binary
                op_0 = [0] * len(OP_SET['4'])
                op_1 = [1] * len(OP_SET['4'])
                op_0[op_id] = 1
                op_1[op_id] = 0
                op_ids += [op_1, op_0]

                # op_ids += [op_id]

                generate_y = self.tokenizer.tokenize(labels) + ['[EOS]']
                gen_ids += [self.tokenizer.convert_tokens_to_ids(generate_y)]

                raw_tgt += [labels]

            # 转变为tensor，其中target需要进行pad，保证统一长度。
            # 例如：[bos,1,2,3,eos] 与 [bos,1,2,eos,pad]
            # 通过loss将pad的影响消去

            input_ids = torch.tensor([f for f in src], dtype=torch.long)
            input_mask = torch.tensor([f for f in input_masks], dtype=torch.long)
            segment_ids = torch.tensor([f for f in segment_ids], dtype=torch.long)
            state_position_ids = torch.tensor([f for f in state_position_ids], dtype=torch.long)
            op_ids = torch.tensor([f for f in op_ids], dtype=torch.long)
            domain_ids = torch.tensor([f for f in domain_ids], dtype=torch.long)
            gen_ids = [b for b in gen_ids]

            max_update = max([len(b) for b in gen_ids])
            max_value = max([b for b in flatten(gen_ids)])
            for bid, b in enumerate(gen_ids):
                n_update = len(b)
                gen_ids[bid] = b + [0] * (max_update - n_update)
            gen_ids = torch.tensor(gen_ids, dtype=torch.long)

            # binary
            op_ids = op_ids.reshape(input_ids.size()[0],-1,len(OP_SET['4'])).permute(0,2,1)


            return {'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'state_position_ids': state_position_ids,
                    'op_ids': op_ids,
                    'domain_ids': domain_ids,
                    'gen_ids': gen_ids,
                    # 'max_value': max_value,
                    'max_update': max_update,
                    'domain2ids': OP_SET['0']
                    }
        else:
            for temp in data:
                state = self.state
                sline = temp["text"].strip()

                # if pad
                avail_src_length = self.max_len - len(state) - 3
                # while len(sline) < self.max_len:
                #     sline = sline + '<pad>'
                #
                # sline = sline[0:self.max_len]

                src_tokens = self.tokenizer.tokenize(' '.join(sline))

                if len(src_tokens) > avail_src_length:
                    # avail_length = len(src_tokens) - avail_src_length
                    src_tokens = src_tokens[:avail_src_length]

                drop_mask = [0] + [0] * len(state) + [0] + [1] * len(src_tokens) + [0]
                diag_1 = ["[CLS]"] + state + ["[SEP]"]
                diag_2 = src_tokens + ["[SEP]"]
                segment = [0] * len(diag_1) + [1] * len(diag_2)

                diag = diag_1 + diag_2
                # word dropout
                if config.getfloat("model", "word_dropout") > 0.:
                    drop_mask = np.array(drop_mask)
                    word_drop = np.random.binomial(drop_mask.astype('int64'), config.getfloat("model", "word_dropout"))
                    diag = [w if word_drop[i] == 0 else '[UNK]' for i, w in enumerate(diag)]

                input_ = diag
                segment_id = segment
                # input_ = diag + state
                # segment_id = segment + [1] * len(state)

                slot_position = []
                for i, t in enumerate(input_):
                    if t == '[SLOT]':
                        slot_position.append(i)

                input_mask = [1] * len(input_)
                input_id = self.tokenizer.convert_tokens_to_ids(input_)
                if len(input_mask) < self.max_len:
                    input_id = input_id + [0] * (self.max_len - len(input_mask))
                    segment_id = segment_id + [0] * (self.max_len - len(input_mask))
                    input_mask = input_mask + [0] * (self.max_len - len(input_mask))

                # append
                src += [input_id]
                segment_ids += [segment_id]
                input_masks += [input_mask]
                state_position_ids += [slot_position]

                # append
                tline = temp["label"].strip()
                tgtWords = tline.split()
                try:
                    labels = tgtWords[2]
                except IndexError as e:
                    labels = tgtWords[1]
                domain_ids += [OP_SET['0'][tgtWords[0]]]
                op_id = OP_SET['4'][tgtWords[1]]
                # binary
                op_0 = [0] * len(OP_SET['4'])
                op_1 = [1] * len(OP_SET['4'])
                op_0[op_id] = 1
                op_1[op_id] = 0
                op_ids += [op_1, op_0]

                # op_ids += [op_id]

                generate_y = self.tokenizer.tokenize(labels) + ['[EOS]']
                gen_ids += [self.tokenizer.convert_tokens_to_ids(generate_y)]

                temp_tgt = self.eval_dict.convertTonumpyIdx([labels], unkWord=False, bosWord=False, eosWord=False)
                raw_tgt += [labels]
                tgt += [temp_tgt]
                tgt_sizes += [len(temp_tgt)]

                # 转变为tensor，其中target需要进行pad，保证统一长度。
                # 例如：[bos,1,2,3,eos] 与 [bos,1,2,eos,pad]
                # 通过loss将pad的影响消去

            input_ids = torch.tensor([f for f in src], dtype=torch.long)
            input_mask = torch.tensor([f for f in input_masks], dtype=torch.long)
            segment_ids = torch.tensor([f for f in segment_ids], dtype=torch.long)
            state_position_ids = torch.tensor([f for f in state_position_ids], dtype=torch.long)
            op_ids = torch.tensor([f for f in op_ids], dtype=torch.long)
            domain_ids = torch.tensor([f for f in domain_ids], dtype=torch.long)
            gen_ids = [b for b in gen_ids]

            max_update = max([len(b) for b in gen_ids])
            max_value = max([b for b in flatten(gen_ids)])
            for bid, b in enumerate(gen_ids):
                n_update = len(b)
                gen_ids[bid] = b + [0] * (max_update - n_update)
            gen_ids = torch.tensor(gen_ids, dtype=torch.long)

            # binary
            op_ids = op_ids.reshape(input_ids.size()[0], -1, len(OP_SET['4'])).permute(0, 2, 1)

            tgt_len = [len(s) for s in tgt]
            tgt_pad = np.zeros((len(tgt), self.eval_dict.size()), dtype=np.int)
            for i, s in enumerate(tgt):
                end = tgt_len[i]
                tar = s[:end]
                temp = np.zeros(self.eval_dict.size())
                indices = [label for label in tar]
                if not None in indices:
                    temp[indices] = 1

                tgt_pad[i, :self.eval_dict.size()] = temp

            tgt = torch.FloatTensor(tgt_pad)

            return {'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'state_position_ids': state_position_ids,
                    'op_ids': op_ids,
                    'domain_ids': domain_ids,
                    'gen_ids': gen_ids,
                    # 'max_value': max_value,
                    'max_update': max_update,
                    'domain2ids': OP_SET['0'],
                    "raw_tgt": raw_tgt,
                    "tgt": tgt
                    }