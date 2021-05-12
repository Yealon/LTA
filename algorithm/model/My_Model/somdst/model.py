"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
from transformers import BertConfig, AutoModelForMaskedLM, BertTokenizer, AlbertForMaskedLM, AlbertModel
from transformers.modeling_bert import BertPreTrainedModel, BertModel

from model.loss import masked_cross_entropy_for_value
from output.My_Model.SOM_output import SOM_output


class SomDST(nn.Module):
    def __init__(self, config, *args, **params):
#     def __init__(self, config, n_op, n_domain, update_id, exclude_domain=False):
        super(SomDST, self).__init__()
# class SomDST(BertPreTrainedModel):
#     def __init__(self, config, n_op, n_domain, update_id, exclude_domain=False):
#         super(SomDST, self).__init__(config)
        self.config = config
        n_op = config.getint("model", "n_op")
        n_domain = config.getint("model", "n_domain")
        update_id = config.getint("model", "update_id")
        exclude_domain = config.getboolean("model", "exclude_domain")

        self.encoder = Encoder(config, n_op, n_domain, update_id, exclude_domain)
        self.encoder.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
        self.encoder.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)

        self.decoder = Decoder(self.encoder.bert.config, self.encoder.bert.embeddings.word_embeddings.weight)
        # re-initialize added special tokens ([SLOT], [EOS])



        # self.apply(self._init_weights)

    def forward(self, data):
        '''
          input_ids, token_type_ids,
                          state_positions, attention_mask,
                          max_value, op_ids=None, max_update=None, teacher=None
          '''
        input_ids = data["input_ids"]
        token_type_ids = data["segment_ids"]
        state_positions = data["state_position_ids"]
        attention_mask = data["input_mask"]
        op_ids = data["op_ids"]
        max_update = data["max_update"]
        gen_ids = data["gen_ids"]
        domain_ids = data["domain_ids"]

        teacher = None

        enc_outputs = self.encoder(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   state_positions=state_positions,
                                   attention_mask=attention_mask,
                                   op_ids=op_ids[:,:,1])
                                   # max_update=max_update)
        # x, decoder_input, encoder_output, hidden, max_len, teacher = None)
        domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output = enc_outputs
        gen_scores = self.decoder(input_ids, decoder_inputs, sequence_output,
                                  pooled_output, max_update, teacher)

        loss_s = nn.CrossEntropyLoss()(state_scores.view(-1, op_ids.shape[1]), op_ids[:, :, 1].max(1)[1])
        # loss_s = nn.BCEWithLogitsLoss()(state_scores, op_ids)
        loss_g = masked_cross_entropy_for_value(gen_scores.contiguous(),
                                                gen_ids.contiguous(),
                                                BertTokenizer.from_pretrained(self.config.get("model", "bert_path")).vocab['[PAD]'])
        loss = loss_s + loss_g
        if self.config.getboolean("model", "exclude_domain") is not True:
            loss_d = nn.CrossEntropyLoss()(domain_scores.view(-1, len(data['domain2ids'])), domain_ids.view(-1))
            loss = loss + loss_d
        # batch_loss.append(loss.item())
        return loss

        # return domain_scores, state_scores, gen_scores



    @SOM_output
    def valid(self, data, mode):
        input_ids = data["input_ids"]
        token_type_ids = data["segment_ids"]
        state_positions = data["state_position_ids"]
        attention_mask = data["input_mask"]
        op_ids = data["op_ids"]
        max_update = data["max_update"]
        gen_ids = data["gen_ids"]
        domain_ids = data["domain_ids"]

        teacher = None

        enc_outputs = self.encoder(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   state_positions=state_positions,
                                   attention_mask=attention_mask
                                   # op_ids=op_ids[:, :, 1]
                                   )
        # max_update=max_update)
        # x, decoder_input, encoder_output, hidden, max_len, teacher = None)
        domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output = enc_outputs
        gen_scores = self.decoder(input_ids, decoder_inputs, sequence_output,
                                  pooled_output, max_update, teacher)

        if 'test' == mode:
            return {
                'mode': mode,
                'y_pre': gen_scores.squeeze(1).max(-1)[1].tolist(),
                'config': self.config,
                'raw_tgt': data['raw_tgt'],
                'y': data["tgt"],
            }
        else:
            return {'mode': mode,
                    'y_pre':gen_scores.squeeze(1).max(-1)[1].tolist(),
                    'y': data["tgt"],
                    'config': self.config,
                    }



        # return domain_scores, state_scores, gen_scores



class Encoder(nn.Module):
    def __init__(self, config, n_op, n_domain, update_id, exclude_domain=False):
        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))
        bert_config = self.bert.config
        bert_config.dropout = config.getfloat("model", "dropout")
        self.hidden_size = bert_config.hidden_size
        self.exclude_domain = exclude_domain


        self.dropout = nn.Dropout(bert_config.dropout)
        self.action_cls = nn.Linear(bert_config.hidden_size, n_op)
        if self.exclude_domain is not True:
            self.domain_cls = nn.Linear(bert_config.hidden_size, n_domain)
        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id

    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask,
                op_ids=None, max_update=None, thres=None):
        bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output, pooled_output = bert_outputs[:2]
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        state_output = torch.gather(sequence_output, 1, state_pos)
        state_scores = torch.nn.functional.softmax(self.action_cls(self.dropout(state_output)).squeeze(dim=2), dim=1)  # B,4_num,2
        if self.exclude_domain:
            domain_scores = torch.zeros(1, device=input_ids.device)  # dummy
        else:
            domain_scores = self.domain_cls(self.dropout(pooled_output))

        batch_size = state_scores.size(0)

        if op_ids is None:
            # op_ids = state_scores.view(-1, self.n_op).max(-1)[-1].view(batch_size, -1)
            op_indexs = state_scores.max(-1)[-1].view(batch_size, -1)
            op_ids = torch.zeros(op_indexs.shape[0], state_pos.shape[1], device=input_ids.device)
            op_ids.scatter_(1, op_indexs, 1)
        if max_update is None:
            max_update = op_ids.eq(self.update_id).sum(-1).max().item()
        # todo
        gathered = []
        for b, a in zip(state_output, op_ids.eq(self.update_id)):  # update
            if a.sum().item() != 0:
                v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.hidden_size)
                n = v.size(1)
                gap = max_update - n
                if gap > 0:
                    zeros = torch.zeros(1, 1*gap, self.hidden_size, device=input_ids.device)
                    v = torch.cat([v, zeros], 1)
            else:
                v = torch.zeros(1, max_update, self.hidden_size, device=input_ids.device)
            gathered.append(v)
        decoder_inputs = torch.cat(gathered)
        return domain_scores, state_scores, decoder_inputs, sequence_output, pooled_output.unsqueeze(0)


class Decoder(nn.Module):
    def __init__(self, bert_config, bert_model_embedding_weights):
        super(Decoder, self).__init__()
        self.pad_idx = 0
        self.hidden_size = bert_config.hidden_size
        self.vocab_size = bert_config.vocab_size
        # bert version
        self.embed = nn.Embedding(bert_config.vocab_size, bert_config.hidden_size, padding_idx=self.pad_idx)
        self.embed.weight = bert_model_embedding_weights
        # albert versionn
        # self.embed = nn.Embedding(bert_config.vocab_size, bert_config.embedding_size, padding_idx=self.pad_idx)
        # self.embed.weight = bert_model_embedding_weights
        # self.hidden_linear = nn.Linear(bert_config.embedding_size, bert_config.hidden_size)
        # bert version
        self.gru = nn.GRU(bert_config.hidden_size, bert_config.hidden_size, 1, batch_first=True)
        # albert versionn
        # self.gru = nn.GRU(bert_config.hidden_size, bert_config.hidden_size, 1, batch_first=True)
        self.w_gen = nn.Linear(bert_config.hidden_size*3, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(bert_config.dropout)

        for n, p in self.gru.named_parameters():
            if 'weight' in n:
                p.data.normal_(mean=0.0, std=bert_config.initializer_range)

    def forward(self, x, decoder_input, encoder_output, hidden, max_len, teacher=None):
        # max_len:decoder端输出最多长度
        mask = x.eq(self.pad_idx)
        batch_size, n_update, _ = decoder_input.size()  # B,J',5 # long
        state_in = decoder_input
        all_point_outputs = torch.zeros(n_update, batch_size, max_len, self.vocab_size).to(x.device)
        result_dict = {}
        for j in range(n_update):  # n_update: encoder端[SEP]预测为1的数量
            w = state_in[:, j].unsqueeze(1)  # B,1,D
            slot_value = []
            for k in range(max_len):
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B,D
                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                attn_e = attn_e.squeeze(-1).masked_fill(mask, -1e9)
                attn_history = nn.functional.softmax(attn_e, -1)  # B,T

                # B,D * D,V => B,V
                # attn_v = torch.matmul(hidden.squeeze(0), self.hidden_linear(self.embed.weight).transpose(0, 1))  # B,V
                attn_v = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V
                attn_vocab = nn.functional.softmax(attn_v, -1)

                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D

                p_gen = self.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1)))  # B,1
                p_gen = p_gen.squeeze(-1)

                p_context_ptr = torch.zeros_like(attn_vocab).to(x.device)
                p_context_ptr.scatter_add_(1, x, attn_history)  # copy B,V
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
                _, w_idx = p_final.max(-1)
                slot_value.append([ww.tolist() for ww in w_idx])
                if teacher is not None:
                    # w = self.hidden_linear(self.embed(teacher[:, j, k]).unsqueeze(1))
                    w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    # w = self.hidden_linear(self.embed(w_idx).unsqueeze(1))  # B,1,D
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D
                all_point_outputs[j, :, k, :] = p_final

        return all_point_outputs.transpose(0, 1)

