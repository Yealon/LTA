import torch
import torch.nn as nn

from model.TextLSTM.Att import Attn
from output.Basic.threshold_onehot import threshold_onehot
from utils.dict_utils import Dict
from torch.autograd import Variable


class ParMatchLSTM(nn.Module):
    """
    BiLSTM: BiLSTM, BiGRU
    """

    def __init__(self, config, *args, **params):
        super(ParMatchLSTM, self).__init__()
        self.config = config

        self.hidden_size = config.getint("model", "hidden_size")

        self.embedding = nn.Embedding(Dict(config.get("model", "src_vocab_path")).size(),
                                      config.getint("model", "emb_size"))

        if config.getboolean("model", "use_gru"):
            self.bilstm = nn.GRU(config.getint("model", "emb_size"),
                                 self.hidden_size,
                                 config.getint("model", "num_layers"),
                                 dropout=(0 if config.getint("model", "num_layers") == 1 else config.getfloat("model",
                                                                                                              "model_dropout")),
                                 bidirectional=True,
                                 batch_first=True)
        else:
            self.bilstm = nn.LSTM(config.getint("model", "emb_size"),
                                  self.hidden_size,
                                  config.getint("model", "num_layers"),
                                  dropout=(0 if config.getint("model", "num_layers") == 1 else config.getfloat("model",
                                                                                                               "model_dropout")),
                                  bidirectional=True,
                                  batch_first=True)


        self.fc = nn.Linear(self.hidden_size, config.getint("model", "n_class_retrg"))

        self.fc_dropout = nn.Dropout(config.getfloat("model", "dropout"))

        self.attn = Attn(self.hidden_size)


        self.context_LSTM = nn.LSTM(
            input_size=config.getint("model", "emb_size"),
            hidden_size=config.getint("model", "hidden_size"),
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # # ----- Matching Layer -----
        # for i in range(1, 3):
        #     setattr(self, f'mp_w{i}',
        #             nn.Parameter(torch.rand(1, config.getint("model", "hidden_size"))))
        self.mp_w1 = nn.Parameter(torch.rand(1, config.getint("model", "hidden_size")))
        self.mp_w2 = nn.Parameter(torch.rand(1, config.getint("model", "hidden_size")))

        # ----- Aggregation Layer -----
        self.aggregation_LSTM = nn.LSTM(
            input_size= 2,  # 2*种类 ,initial 8
            hidden_size= config.getint("model", "hidden_size"),
            num_layers= 1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Prediction Layer -----
        self.pred_fc = nn.Linear(config.getint("model", "hidden_size") * 4,
                                  config.getint("model", "hidden_size") * 2)
        self.pred_fc1 = nn.Linear(config.getint("model", "hidden_size") * 2,
                                  config.getint("model", "n_class_dummy_1"))
        self.pred_fc2 = nn.Linear(config.getint("model", "hidden_size") * 2,
                                  config.getint("model", "n_class_dummy_2"))

        self.pred_fc3 = nn.Linear(config.getint("model", "hidden_size") * 2,
                                  config.getint("model", "n_class_dummy_3"))


    # ----- Matching Layer -----

    def dropout(self, v):
        return torch.nn.functional.dropout(v, p=self.config.getfloat("model", "dropout"), training=self.training)

    def mp_matching_func(self, v1, v2, w):
        """
        :param v1: (batch, seq_len, hidden_size)
        :param v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
        :param w: (l, hidden_size)
        :return: (batch, l)
        """
        seq_len = v1.size(1)

        # Trick for large memory requirement
        """
        if len(v2.size()) == 2:
            v2 = torch.stack([v2] * seq_len, dim=1)

        m = []
        for i in range(self.l):
            # v1: (batch, seq_len, hidden_size)
            # v2: (batch, seq_len, hidden_size)
            # w: (1, 1, hidden_size)
            # -> (batch, seq_len)
            m.append(F.cosine_similarity(w[i].view(1, 1, -1) * v1, w[i].view(1, 1, -1) * v2, dim=2))

        # list of (batch, seq_len) -> (batch, seq_len, l)
        m = torch.stack(m, dim=2)
        """

        # (1, 1, hidden_size, l)
        w = w.transpose(1, 0).unsqueeze(0).unsqueeze(0)
        # (batch, seq_len, hidden_size, l)
        v1 = w * torch.stack([v1], dim=3)
        if len(v2.size()) == 3:
            v2 = w * torch.stack([v2], dim=3)
        else:
            v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)], dim=3)

        m = torch.nn.functional.cosine_similarity(v1, v2, dim=2)

        return m

    def match(self, text, label_meaning, n):
        p = self.embedding(text)
        h = self.embedding(label_meaning)

        # ----- Context Representation Layer -----
        # (batch, seq_len, hidden_size * 2)
        con_p, _ = self.context_LSTM(p)
        con_h, _ = self.context_LSTM(h)

        con_p = self.dropout(con_p)
        con_h = self.dropout(con_h)

        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = torch.split(con_p, self.config.getint("model", "hidden_size"), dim=-1)
        con_h_fw, con_h_bw = torch.split(con_h, self.config.getint("model", "hidden_size"), dim=-1)

        # 1. Full-Matching

        # (batch, seq_len, hidden_size), (batch, hidden_size)
        # -> (batch, seq_len, l)
        mv_p_full_fw = self.mp_matching_func(con_p_fw, con_h_fw[:, -1, :], self.mp_w1)
        mv_p_full_bw = self.mp_matching_func(con_p_bw, con_h_bw[:, 0, :], self.mp_w2)
        mv_h_full_fw = self.mp_matching_func(con_h_fw, con_p_fw[:, -1, :], self.mp_w1)
        mv_h_full_bw = self.mp_matching_func(con_h_bw, con_p_bw[:, 0, :], self.mp_w2)

        mv_p = torch.cat(
            [mv_p_full_fw, mv_p_full_bw], dim=2)
        mv_h = torch.cat(
            [mv_h_full_fw, mv_h_full_bw], dim=2)

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        # ----- Aggregation Layer -----
        # (batch, seq_len, l * 8) -> (2, batch, hidden_size)
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)

        # 2 * (2, batch, hidden_size) -> 2 * (batch, hidden_size * 2) -> (batch, hidden_size * 4)
        x = torch.cat(
            [agg_p_last.permute(1, 0, 2).contiguous().view(-1, self.config.getint("model", "hidden_size") * 2),
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.config.getint("model", "hidden_size") * 2)],
            dim=1)
        x = self.dropout(x)

        # ----- Prediction Layer -----
        x = torch.tanh(self.pred_fc(x))
        x = self.dropout(x)
        if 1 == n:
            x = self.pred_fc1(x)
        elif 2 == n:
            x = self.pred_fc2(x)
        else:
            x = self.pred_fc3(x)
        return x

    def forward(self, data):

        """
        :param x: [batch_size, max_len]
        :return logits: logits
        """
        # self.bilstm.module.flatten_parameters()
        # self.context_LSTM.module.flatten_parameters()
        # self.aggregation_LSTM.module.flatten_parameters()

        # self.bilstm.flatten_parameters()
        # if not hasattr(self, '_flattened'):
        #     self.bilstm.flatten_parameters()
        # setattr(self, '_flattened', True)

        x = data['src_pad']
        # tgt = data["tgt"]
        dum_x = data["dum_src_pad"]
        rest_tgt = data["rest_tgt_pad"]
        dummy_tgt = data['dummy_tgt_pad']
        meaning_pad = data['meaning_pad']
        level12 = data['level12']
        level23 = data['level23'] # level:4,6,13,独热编码为n（4/6/13），26

        x = self.embedding(x)  # (batch_size, max_len, word_vec)
        # 输入的x是所有time step的输入, 输出的y实际每个time step的hidden输出
        # _是最后一个time step的hidden输出
        # 因为双向,y的shape为(batch_size, max_len, hidden_size*num_directions),
        # 其中[:,:,:hidden_size]是前向的结果,[:,:,hidden_size:]是后向的结果
        y, _ = self.bilstm(x)  # (batch_size, max_len, hidden_size*num_directions)
        y = y[:, :, :self.hidden_size] + y[:, :, self.hidden_size:]  # (batch_size, max_len, hidden_size)
        alpha = self.attn(y)  # (batch_size, 1, max_len)
        r = alpha.bmm(y).squeeze(1)  # (batch_size, hidden_size)
        h = torch.tanh(r)  # (batch_size, hidden_size)
        large_logits = self.fc(h)  # (batch_size, class_num)
        large_logits = self.fc_dropout(large_logits)

        # start ############少数样本匹配####################

        # 按层次对少数样本进行匹配

        sr_len = dum_x.size(0)
        s_logits = Variable(
            torch.Tensor(sr_len, Dict(self.config.get("model", "dummy_vocab_path")).size()).fill_(-1e7).cuda())

        if sr_len > 0:

            match_meaning = meaning_pad[0:4, :].repeat(1, sr_len, 1).reshape(-1, meaning_pad.size(1))
            match_dum = dum_x.repeat(4, 1, 1).permute(1, 0, 2).reshape(-1, dum_x.size(1))

            first_all_outputs, _ = self.match(match_dum, match_meaning, 1).max(1)
            first_outputs = first_all_outputs.reshape(sr_len, -1)
            _, outputs1_index = first_outputs.max(dim=1)

            s_logits[list(range(sr_len)), 0:4] = first_outputs

            # 第二层
            max_2_index = []
            # ori_2_index = []
            select2 = torch.index_select(level12, 0, outputs1_index)
            select2 = torch.nonzero(select2)
            # select2：一系列坐标，第二层的相应index：
            select2 = select2.t()
            match_meaning_2 = meaning_pad[select2[1]]
            match_dum_2 = dum_x[select2[0]]

            second_all_outputs = self.match(match_dum_2, match_meaning_2, 2)
            second_outputs, _ = second_all_outputs.max(1)
            # second_outputs = second_outputs.squeeze(1)

            for i in range(0, outputs1_index.size(0)):

                mask_child = (torch.ne(select2[0], i).int() == 1).nonzero().squeeze(1)
                if mask_child.size(0) == select2[0].size(0):
                    continue
                second_child_scores = torch.sigmoid(second_outputs.scatter(dim=0, index=mask_child, value=-1e7))
                _, max_index = second_child_scores.max(dim=0)
                s_logits[i, 4:13] = second_all_outputs[max_index]
                # ori_2_index.append(max_index)
                max_2_index.append(select2[1][max_index])

            # ori_2_index = torch.stack(ori_2_index)
            max_2_index = torch.stack(max_2_index)
            # max_2_index = select2[1][ori_2_index]
            # second_all_outputs[temp_index]
            # outputs2_index = select2[1][max_2_index]
            # outputs2 = second_outputs[max_2_index]

            # 第三层
            # max_3_index = []
            select3 = torch.index_select(level23, 0, max_2_index - 4)
            select3 = torch.nonzero(select3)
            # select2：一系列坐标，第二层的相应index：
            select3 = select3.t()

            match_meaning_3 = meaning_pad[select3[1]]
            match_dum_3 = dum_x[select3[0]]

            third_all_outputs = self.match(match_dum_3, match_meaning_3, 3)
            third_outputs, _ = third_all_outputs.max(1)
            # third_outputs = third_outputs.squeeze(1)

            for i in range(0, max_2_index.size(0)):
                mask_child = (torch.ne(select3[0], i).int() == 1).nonzero().squeeze(1)
                if mask_child.size(0) == select3[0].size(0):
                    continue
                third_child_scores = torch.sigmoid(third_outputs.scatter(dim=0, index=mask_child, value=-1e7))

                _, max_index = third_child_scores.max(dim=0)
                s_logits[i, 13:] = third_all_outputs[max_index]
                # max_3_index.append(select3[1][max_index])
            # max_3_index = torch.stack(max_3_index)
            # # outputs3_index = select3[1][max_3_index] # 5 12 8 / 15 25 20
            # # outputs3 = third_outputs[max_3_index]

            criterion1 = nn.BCEWithLogitsLoss()
            loss1 = criterion1(large_logits, rest_tgt)
            criterion2 = nn.BCEWithLogitsLoss()
            loss2 = criterion2(s_logits, dummy_tgt)

            total_loss = loss2 + loss1
            # total_loss.backward()
            # total_loss = total_loss.item()
            return total_loss

        else:
            criterion1 = nn.BCEWithLogitsLoss()
            total_loss = criterion1(large_logits, rest_tgt)
            # total_loss.backward()
            # total_loss = total_loss.item()
            return total_loss

    @threshold_onehot
    def valid(self, data, mode):

        self.bilstm.flatten_parameters()
        if not hasattr(self, '_flattened'):
            self.bilstm.flatten_parameters()
        setattr(self, '_flattened', True)

        x = data['src_pad']
        meaning_pad = data['meaning_pad']
        level12 = data['level12']
        level23 = data['level23'] # l

        x = self.embedding(x)  # (batch_size, max_len, word_vec)
        # 输入的x是所有time step的输入, 输出的y实际每个time step的hidden输出
        # _是最后一个time step的hidden输出
        # 因为双向,y的shape为(batch_size, max_len, hidden_size*num_directions),
        # 其中[:,:,:hidden_size]是前向的结果,[:,:,hidden_size:]是后向的结果
        y, _ = self.bilstm(x)  # (batch_size, max_len, hidden_size*num_directions)
        y = y[:, :, :self.hidden_size] + y[:, :, self.hidden_size:]  # (batch_size, max_len, hidden_size)
        alpha = self.attn(y)  # (batch_size, 1, max_len)
        r = alpha.bmm(y).squeeze(1)  # (batch_size, hidden_size)
        h = torch.tanh(r)  # (batch_size, hidden_size)
        large_logits = self.fc(h)  # (batch_size, class_num)
        large_logits = self.fc_dropout(large_logits)

        # start ############少数样本匹配####################
        l_scores = torch.sigmoid(large_logits)
        dum_scores = l_scores[:, 4]
        dum_index = (torch.gt(dum_scores, self.config.getfloat("output", "threshold")).int() == 1).nonzero().squeeze(1).cuda()
        dum_x = data['src_pad'].index_select(0, dum_index)
        # large_logits[dum_index, :] = large_logits[dum_index, :].fill_(-1e7).cuda()
        s_logits = Variable(
            torch.Tensor(x.size(0), Dict(self.config.get("model", "dummy_vocab_path")).size()).fill_(-1e7).cuda())

        # 按层次对少数样本进行匹配

        sr_len = dum_x.size(0)

        if sr_len > 0:

            # outputs = []
            # output = None
            match_meaning = meaning_pad[0:4, :].repeat(1, sr_len, 1).reshape(-1, meaning_pad.size(1))
            match_dum = dum_x.repeat(4, 1, 1).permute(1, 0, 2).reshape(-1, dum_x.size(1))

            # match_meaning = meaning_pad[0:4, :].repeat(sr_len, 1, 1).permute(1,0,2).reshape(-1,meaning_pad.size(1))
            # match_dum = dum_x.repeat(1, 4, 1).reshape(-1, dum_x.size(1))
            # first_outputs, _ = self.match(match_dum, match_meaning, 1).max(1)
            # outputs1, outputs1_index = first_outputs.reshape(sr_len, -1).max(dim=1)

            first_all_outputs, _ = self.match(match_dum, match_meaning, 1).max(1)
            first_outputs = first_all_outputs.reshape(sr_len, -1)
            _, outputs1_index = first_outputs.max(dim=1)

            s_logits[dum_index, 0:4] = first_outputs

            # s_logits[dum_index, outputs1_index] = outputs1

            # 第二层
            max_2_index = []
            select2 = torch.index_select(level12, 0, outputs1_index)
            select2 = torch.nonzero(select2)
            # select2：一系列坐标，第二层的相应index：
            select2 = select2.t()
            match_meaning_2 = meaning_pad[select2[1]]
            match_dum_2 = dum_x[select2[0]]

            # second_outputs, _ = self.match(match_dum_2, match_meaning_2, 2).max(1)

            second_all_outputs = self.match(match_dum_2, match_meaning_2, 2)
            second_outputs, _ = second_all_outputs.max(1)

            for n, i in enumerate(list(dum_index)):
                mask_child = (torch.ne(select2[0], n).int() == 1).nonzero().squeeze(1)
                if mask_child.size(0) == select2[0].size(0):
                    continue
                # second_child_scores = torch.sigmoid(second_outputs.scatter(dim=0, index=mask_child, value=-1e7))
                # _, max_index = second_child_scores.max(dim=0)
                # s_logits[i, select2[1][max_index]] = second_outputs[max_index]
                second_child_scores = torch.sigmoid(second_outputs.scatter(dim=0, index=mask_child, value=-1e7))
                _, max_index = second_child_scores.max(dim=0)
                s_logits[i, 4:13] = second_all_outputs[max_index]

                max_2_index.append(select2[1][max_index])
            max_2_index = torch.stack(max_2_index)

            # 第三层
            # max_3_index = []
            select3 = torch.index_select(level23, 0, max_2_index - 4)
            select3 = torch.nonzero(select3)
            # select2：一系列坐标，第二层的相应index：
            select3 = select3.t()

            match_meaning_3 = meaning_pad[select3[1]]
            match_dum_3 = dum_x[select3[0]]

            # third_outputs, _ = self.match(match_dum_3, match_meaning_3, 3).max(1)

            third_all_outputs = self.match(match_dum_3, match_meaning_3, 3)
            third_outputs, _ = third_all_outputs.max(1)

            for n, i in enumerate(list(dum_index)):
                mask_child = (torch.ne(select3[0], n).int() == 1).nonzero().squeeze(1)
                if mask_child.size(0) == select3[0].size(0):
                    continue
                third_child_scores = torch.sigmoid(third_outputs.scatter(dim=0, index=mask_child, value=-1e7))

                _, max_index = third_child_scores.max(dim=0)
                s_logits[i, 13:] = third_all_outputs[max_index]

                # third_child_scores = torch.sigmoid(third_outputs.scatter(dim=0, index=mask_child, value=-1e7))
                #
                # _, max_index = third_child_scores.max(dim=0)
                # s_logits[i, select3[1][max_index]] = third_outputs[max_index]

            # combine three levels

        total_logit = torch.cat(
            [large_logits[:, 0:4], s_logits[:, 0:4],  # 6
             large_logits[:, 5:10], s_logits[:, 4:13],  # 14
             large_logits[:, 11:48], s_logits[:, 13:]], dim=1)  # 37+13

        if 'test' == mode:
            return {
                'mode': mode,
                'y_pre': torch.sigmoid(total_logit),
                'config': self.config,
                'raw_tgt': data['raw_tgt'],
                'y': data["tgt"],
            }
        else:
            return {'mode': mode,
                    'y_pre': torch.sigmoid(total_logit),
                    'y': data["tgt"],
                    'config': self.config,
                    }



        # else:
        #     total_logit = torch.cat(
        #         [large_logits[:, 0:4], s_logits[:, 0:4],
        #          large_logits[:, 5:10], s_logits[:, 4:13],
        #          large_logits[:, 11:48], s_logits[:, 13:]], dim=1)
        #     return torch.sigmoid(total_logit)




