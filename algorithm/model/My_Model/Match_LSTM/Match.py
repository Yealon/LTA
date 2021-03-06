import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dict_utils import Dict


class Match(nn.Module):
    def __init__(self, config, n_class):

        super(Match, self).__init__()

        self.config = config
        self.d = config.getint("model", "emb_size")
        # todo
        self.l = 1
        # todo
        self.word_emb = nn.Embedding(Dict(config.get("model", "src_vocab_path")).size(),
                                     config.getint("model", "emb_size"))

        # ----- Context Representation Layer -----
        self.context_LSTM = nn.LSTM(
            input_size=self.d,
            hidden_size=config.getint("model", "hidden_size"),
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Matching Layer -----
        for i in range(0, 3):
            setattr(self, f'mp_w{i}',
                    nn.Parameter(torch.rand(self.l, self.config.getint("model", "hidden_size"))))

        # ----- Aggregation Layer -----
        self.aggregation_LSTM = nn.LSTM(
            input_size=self.l * 2,  # 2*种类 ,initial 8
            hidden_size=self.config.getint("model", "hidden_size"),
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Prediction Layer -----
        self.pred_fc1 = nn.Linear(self.config.getint("model", "hidden_size") * 4,
                                  self.config.getint("model", "hidden_size") * 2)
        self.pred_fc2 = nn.Linear(self.config.getint("model", "hidden_size") * 2,
                                  n_class)
                                  # self.config.getint("model", "n_class_dummy"))

        self.reset_parameters()

    def reset_parameters(self):
        # ----- Word Representation Layer -----

        # ----- Context Representation Layer -----
        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0)
        nn.init.constant_(self.context_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0)
        nn.init.constant_(self.context_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.context_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.context_LSTM.bias_hh_l0_reverse, val=0)

        # ----- Matching Layer -----
        for i in range(0, 3):
            w = getattr(self, f'mp_w{i}')
            nn.init.kaiming_normal_(w)

        # ----- Aggregation Layer -----
        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.aggregation_LSTM.weight_ih_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.aggregation_LSTM.weight_hh_l0_reverse)
        nn.init.constant_(self.aggregation_LSTM.bias_hh_l0_reverse, val=0)

        # ----- Prediction Layer ----
        nn.init.uniform_(self.pred_fc1.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc1.bias, val=0)

        nn.init.uniform_(self.pred_fc2.weight, -0.005, 0.005)
        nn.init.constant_(self.pred_fc2.bias, val=0)

    # def init_multi_gpu(self, device, config, *args, **params):
    #     self.word_emb = nn.DataParallel(self.word_emb, device_ids=device)
    #     self.context_LSTM = nn.DataParallel(self.context_LSTM, device_ids=device)
    #     self.pred_fc1 = nn.DataParallel(self.pred_fc1, device_ids=device)
    #     self.pred_fc2 = nn.DataParallel(self.pred_fc2, device_ids=device)
    #



    def dropout(self, v):
        return F.dropout(v, p=self.config.getfloat("model", "dropout"), training=self.training)

    def forward(self, label_meaning, text, config, gpu_list, mode):

        self.context_LSTM.flatten_parameters()
        if not hasattr(self, '_flattened'):
            self.context_LSTM.flatten_parameters()
        setattr(self, '_flattened', True)

        self.aggregation_LSTM.flatten_parameters()
        if not hasattr(self, '_flattened'):
            self.aggregation_LSTM.flatten_parameters()
        setattr(self, '_flattened', True)


        # ----- Matching Layer -----
        def mp_matching_func(v1, v2, w):
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
            v1 = w * torch.stack([v1] * self.l, dim=3)
            if len(v2.size()) == 3:
                v2 = w * torch.stack([v2] * self.l, dim=3)
            else:
                v2 = w * torch.stack([torch.stack([v2] * seq_len, dim=1)] * self.l, dim=3)

            m = F.cosine_similarity(v1, v2, dim=2)

            return m

        def mp_matching_func_pairwise(v1, v2, w):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size)
            :param w: (l, hidden_size)
            :return: (batch, l, seq_len1, seq_len2)
            """

            # Trick for large memory requirement
            """
            m = []
            for i in range(self.l):
                # (1, 1, hidden_size)
                w_i = w[i].view(1, 1, -1)
                # (batch, seq_len1, hidden_size), (batch, seq_len2, hidden_size)
                v1, v2 = w_i * v1, w_i * v2
                # (batch, seq_len, hidden_size->1)
                v1_norm = v1.norm(p=2, dim=2, keepdim=True)
                v2_norm = v2.norm(p=2, dim=2, keepdim=True)

                # (batch, seq_len1, seq_len2)
                n = torch.matmul(v1, v2.permute(0, 2, 1))
                d = v1_norm * v2_norm.permute(0, 2, 1)

                m.append(div_with_small_value(n, d))

            # list of (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, l)
            m = torch.stack(m, dim=3)
            """

            # (1, l, 1, hidden_size)
            w = w.unsqueeze(0).unsqueeze(2)
            # (batch, l, seq_len, hidden_size)
            v1, v2 = w * torch.stack([v1] * self.l, dim=1), w * torch.stack([v2] * self.l, dim=1)
            # (batch, l, seq_len, hidden_size->1)
            v1_norm = v1.norm(p=2, dim=3, keepdim=True)
            v2_norm = v2.norm(p=2, dim=3, keepdim=True)

            # (batch, l, seq_len1, seq_len2)
            n = torch.matmul(v1, v2.transpose(2, 3))
            d = v1_norm * v2_norm.transpose(2, 3)

            # (batch, seq_len1, seq_len2, l)
            m = div_with_small_value(n, d).permute(0, 2, 3, 1)

            return m

        def attention(v1, v2):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size)
            :return: (batch, seq_len1, seq_len2)
            """

            # (batch, seq_len1, 1)
            v1_norm = v1.norm(p=2, dim=2, keepdim=True)
            # (batch, 1, seq_len2)
            v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

            # (batch, seq_len1, seq_len2)
            a = torch.bmm(v1, v2.permute(0, 2, 1))
            d = v1_norm * v2_norm

            return div_with_small_value(a, d)

        def div_with_small_value(n, d, eps=1e-8):
            # too small values are replaced by 1e-8 to prevent it from exploding.
            d = d * (d > eps).float() + eps * (d <= eps).float()
            return n / d

        # ----- Word Representation Layer -----
        # (batch, seq_len) -> (batch, seq_len, word_dim)

        p = self.word_emb(text)
        h = self.word_emb(label_meaning)

        p = self.dropout(p)
        h = self.dropout(h)

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
        mv_p_full_fw = mp_matching_func(con_p_fw, con_h_fw[:, -1, :], self.mp_w1)
        mv_p_full_bw = mp_matching_func(con_p_bw, con_h_bw[:, 0, :], self.mp_w2)
        mv_h_full_fw = mp_matching_func(con_h_fw, con_p_fw[:, -1, :], self.mp_w1)
        mv_h_full_bw = mp_matching_func(con_h_bw, con_p_bw[:, 0, :], self.mp_w2)

        # # 2. Maxpooling-Matching
        #
        # # (batch, seq_len1, seq_len2, l)
        # mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3)
        # mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4)
        #
        # # (batch, seq_len, l)
        # mv_p_max_fw, _ = mv_max_fw.max(dim=2)
        # mv_p_max_bw, _ = mv_max_bw.max(dim=2)
        # mv_h_max_fw, _ = mv_max_fw.max(dim=1)
        # mv_h_max_bw, _ = mv_max_bw.max(dim=1)

        # # 3. Attentive-Matching
        #
        # # (batch, seq_len1, seq_len2)
        # att_fw = attention(con_p_fw, con_h_fw)
        # att_bw = attention(con_p_bw, con_h_bw)
        #
        # # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # # -> (batch, seq_len1, seq_len2, hidden_size)
        # att_h_fw = con_h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        # att_h_bw = con_h_bw.unsqueeze(1) * att_bw.unsqueeze(3)
        # # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # # -> (batch, seq_len1, seq_len2, hidden_size)
        # att_p_fw = con_p_fw.unsqueeze(2) * att_fw.unsqueeze(3)
        # att_p_bw = con_p_bw.unsqueeze(2) * att_bw.unsqueeze(3)
        #
        # # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
        # att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        # att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))
        #
        # # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
        # att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=1), att_fw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        # att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=1), att_bw.sum(dim=1, keepdim=True).permute(0, 2, 1))
        #
        # # (batch, seq_len, l)
        # mv_p_att_mean_fw = mp_matching_func(con_p_fw, att_mean_h_fw, self.mp_w5)
        # mv_p_att_mean_bw = mp_matching_func(con_p_bw, att_mean_h_bw, self.mp_w6)
        # mv_h_att_mean_fw = mp_matching_func(con_h_fw, att_mean_p_fw, self.mp_w5)
        # mv_h_att_mean_bw = mp_matching_func(con_h_bw, att_mean_p_bw, self.mp_w6)

        # # 4. Max-Attentive-Matching
        #
        # # (batch, seq_len1, hidden_size)
        # att_max_h_fw, _ = att_h_fw.max(dim=2)
        # att_max_h_bw, _ = att_h_bw.max(dim=2)
        # # (batch, seq_len2, hidden_size)
        # att_max_p_fw, _ = att_p_fw.max(dim=1)
        # att_max_p_bw, _ = att_p_bw.max(dim=1)
        #
        # # (batch, seq_len, l)
        # mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        # mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        # mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        # mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)
        #
        # # (batch, seq_len, l * 8)
        # mv_p = torch.cat(
        #     [mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
        #      mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], dim=2)
        # mv_h = torch.cat(
        #     [mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
        #      mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], dim=2)

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
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, self.config.getint("model", "hidden_size") * 2)], dim=1)
        x = self.dropout(x)

        # ----- Prediction Layer -----
        x = torch.tanh(self.pred_fc1(x))
        x = self.dropout(x)
        x = self.pred_fc2(x)

        return x
