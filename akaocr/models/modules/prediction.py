import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device, beam_size):
        super(Attention, self).__init__()
        self.device = device
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)
        self.beam_size = int(beam_size)

    def _char_to_one_hot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(self.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, batch_H, text, is_train=True, max_label_length=25):
        """
        Parameters
        ----------
        batch_H : Tensor
            contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_classes]
        text : Tensor
            the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO]
        is_train : bool, optional, default: True
            whether it is training phase or not
        max_label_length : int, optional, default: 25
            max length of text label in the batch

        Returns
        ------
        Tensor
            probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = batch_H.size(0)
        num_steps = max_label_length + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(self.device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device),
                  torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(self.device))

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_one_hots = self._char_to_one_hot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_one_hots)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)  # batch_size x num_steps x num_classes
        else:
            if self.beam_size == 1:
                targets = torch.LongTensor(batch_size).fill_(0).to(self.device)  # [GO] token
                probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(self.device)

                for i in range(num_steps):
                    char_one_hots = self._char_to_one_hot(targets, onehot_dim=self.num_classes)
                    hidden, _ = self.attention_cell(hidden, batch_H, char_one_hots)  # batch_size x hidden_size
                    probs_step = self.generator(hidden[0])  # batch_size x num_classes
                    probs[:, i, :] = probs_step
                    _, next_input = probs_step.max(1)
                    targets = next_input
            elif self.beam_size > 1:
                probs = torch.FloatTensor(batch_size, self.beam_size, num_steps, self.num_classes).fill_(0).to(self.device)

                # beam search to find the best sentence
                for idx in range(batch_size):
                    hidden = (torch.FloatTensor(1, self.hidden_size).fill_(0).to(self.device),
                              torch.FloatTensor(1, self.hidden_size).fill_(0).to(self.device))
                    batch_H_idx = batch_H[idx].unsqueeze(0)
                    sequences = [[[0], 0.0, hidden]]  # beam_size x (num_steps + 1) x 2
                    for _ in range(num_steps):
                        all_candidates = list()
                        for k in range(len(sequences)):
                            seq, score, hidden = sequences[k]
                            torch_seq = torch.LongTensor([seq[-1]]).to(self.device)
                            char_one_hots = self._char_to_one_hot(torch_seq, onehot_dim=self.num_classes)
                            hidden, _ = self.attention_cell(hidden, batch_H_idx, char_one_hots)
                            probs_step = self.generator(hidden[0])  # 1 x num_classes
                            sm_probs_step = torch.nn.Softmax(dim=1)(probs_step)[0].cpu().numpy()  # num_classes

                            best_n = np.argsort(sm_probs_step)[::-1][:self.beam_size]  # top n character ids
                            for j in best_n:
                                candidate = [seq + [j], score + np.log(sm_probs_step[j] + 1e-9), hidden]  # score < 0
                                all_candidates.append(candidate)

                        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
                        sequences = ordered[:self.beam_size]

                    for k in range(len(sequences)):
                        seq, score, _ = sequences[k]
                        seq = seq[1:]
                        for j in range(num_steps):
                            probs[idx, k, j, seq[j]] = score  # negative at best char, 0 at others
            else:
                raise ValueError(f'beam size = {self.beam_size} is invalid.')
        return probs  # batch_size x [beam_size x] num_steps x num_classes


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_one_hots):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_one_hots], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
