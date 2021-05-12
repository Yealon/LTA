from output.Seq.seq_index import seq_index
from .Encoder import rnn_encoder
from .Decoder import rnn_decoder

from model.loss import *
from .Beam_Search import Beam

import utils.dict_utils as Dict_F
from utils.dict_utils import Dict


class Seq2Seq_Att(nn.Module):

    def __init__(self, config, *args, **params):
        super(Seq2Seq_Att, self).__init__()
        self.config = config

        self.src_vocab_size = Dict(config.get("model", "src_vocab_path")).size()

        self.tgt_vocab_size = Dict(config.get("model", "trg_vocab_path")).size()

        self.encoder = rnn_encoder(config, self.src_vocab_size)
        self.decoder = rnn_decoder(config, self.tgt_vocab_size)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, data):
        src_pad = data['src_pad']
        src_len = data['src_len']
        tgt_pad = data["tgt_pad"]
        lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)
        src = torch.index_select(src_pad, dim=0, index=indices)
        tgt = torch.index_select(tgt_pad, dim=0, index=indices)

        src = src.t()
        target = tgt[:, :-1].t()
        contexts, state = self.encoder(src, lengths.tolist())

        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)

        outputs = []
        output = None

        for input in target.split(1):
            output, state, _ = self.decoder(input.squeeze(0), state, output)
            outputs.append(output)
        outputs = torch.stack(outputs)

        label = tgt[:, 1:].t()
        # loss = self.criterion(y, label)
        loss = Seq2Seq_loss(outputs, label)

        return loss

    @seq_index
    def valid(self, data, mode):
        src_pad = data['src_pad']
        src_len = data['src_len']
        if self.config.getint("model", "beam_size") > 1 and (not self.config.getboolean("model", "global_emb")):
            samples = self.beam_sample(src_pad, src_len, beam_size=self.config.getint("model", "beam_size"))
        else:
            samples = self.sample(src_pad, src_len)

        if 'test' == mode:
            return {
                'mode': mode,
                'y_pre': samples,
                'config': self.config,
                'raw_tgt': data['raw_tgt'],
                'y': data["tgt"],
            }
        else:
            return {'mode': mode,
                    'y_pre': samples,
                    'y': data["tgt"],
                    'config': self.config,
                    }



    def sample(self, src, src_len):

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, reverse_indices = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        bos = torch.ones(src.size(0)).long().fill_(Dict_F.BOS)
        src = src.t()

        bos = bos.cuda()

        contexts, state = self.encoder(src, lengths.tolist())

        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)

        inputs, outputs = [bos], []
        # attn_matrix = []
        output = None

        for i in range(self.config.getint("model", "max_time_step")):
            output, state, attn_weights = self.decoder(inputs[i], state, output, outputs)
            predicted = output.max(1)[1]
            inputs += [predicted]
            outputs += [predicted]
            # attn_matrix += [attn_weights]

        outputs = torch.stack(outputs)
        sample_ids = torch.index_select(outputs, dim=1, index=reverse_indices).t()

        # attn_matrix = torch.stack(attn_matrix)
        # alignments = attn_matrix.max(2)[1]
        # alignments = torch.index_select(alignments, dim=1, index=reverse_indices).t()

        return sample_ids.cpu().numpy().tolist()


    def beam_sample(self, src, src_len, beam_size=1, attention_weight=False):
        # (1) Run the encoder on the src.

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = torch.index_select(src, dim=0, index=indices)
        src = src.t()
        batch_size = src.size(1)
        contexts, encState = self.encoder(src, lengths.tolist())

        #  (1b) Initialize for the decoder.
        def var(a):
            return a.clone().detach()
            #.requires_grad_(False)
            # return torch.tensor(a, requires_grad=False)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        contexts = rvar(contexts)

        decState = (rvar(encState[0]), rvar(encState[1]))
        beam = [Beam(beam_size, n_best=1, length_norm=self.config.getboolean("model", "length_norm"))
                for __ in range(batch_size)]
        if self.decoder.attention is not None:
            self.decoder.attention.init_context(contexts)

        # (2) run the decoder to generate sentences, using beam search.

        for i in range(self.config.getint("model", "max_time_step")):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))

            # Run one step.
            output, decState, attn = self.decoder(inp, decState)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
            # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output[:, j], attn[:, j])
                b.beam_update(decState, j)

        # (3) Package everything up.
        allHyps, allScores = [], []
        if attention_weight:
            allWeight, allAttn = [], []

        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps = []
            if attention_weight:
                weight, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)

                if attention_weight:
                    attn.append(att.max(1)[1])
                    weight.append(att)
            allHyps.append(hyps[0])
            allScores.append(scores[0])

            if attention_weight:
                allAttn.append(attn[0])
                allWeight.append(weight[0])

        if attention_weight:
            return allHyps, allAttn, allWeight

        return allHyps
