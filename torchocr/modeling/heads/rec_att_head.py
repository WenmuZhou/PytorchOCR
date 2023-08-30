import torch
import torch.nn as nn
import torch.nn.functional as F

    
class AttentionHead(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionHead, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = AttentionGRUCell(in_channels, hidden_size, out_channels)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char.long(), onehot_dim)
        return input_ont_hot

    def forward(self, inputs, data=None, batch_max_length=25):
        batch_size = inputs.size(0)
        num_steps = batch_max_length

        hidden = (torch.zeros(batch_size, self.hidden_size, device=inputs.device),
                  torch.zeros(batch_size, self.hidden_size, device=inputs.device))
        output_hiddens = []

        if self.training:
            targets = data[0]
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                output_hiddens.append(torch.unsqueeze(hidden[0], axis=1))
            output = torch.cat(output_hiddens, dim=1)
            probs = self.generator(output)
        else:
            targets = torch.zeros(batch_size, dtype=torch.int32, device=inputs.device)
            probs = torch.zeros(batch_size, num_steps, self.num_classes, device=inputs.device)
            # probs = None
            char_onehots = None
            outputs = None
            alpha = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs, char_onehots)
                probs_step = self.generator(hidden[0])

                probs[:, i, :] = probs_step
                # if probs is None:
                #     probs = torch.unsqueeze(probs_step, dim=1)
                # else:
                #     probs = torch.concat(
                #         [probs, torch.unsqueeze(
                #             probs_step, dim=1)], dim=1)
                next_input = probs_step.argmax(dim=1)
                targets = next_input
        if not self.training:
            probs = F.softmax(probs, dim=2)
        return {'res':probs}

class AttentionLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionLSTM, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = AttentionLSTMCell(
            in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, data=None, batch_max_length=25):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length

        hidden = (torch.zeros((batch_size, self.hidden_size)), torch.zeros(
            (batch_size, self.hidden_size)))
        output_hiddens = []

        if self.training:
            targets = data[0]
            for i in range(num_steps):
                # one-hot vectors for a i-th char
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs,
                                                    char_onehots)

                hidden = (hidden[1][0], hidden[1][1])
                output_hiddens.append(torch.unsqueeze(hidden[0], dim=1))
            output = torch.cat(output_hiddens, dim=1)
            probs = self.generator(output)

        else:
            targets = torch.zeros(batch_size, device=inputs.device, dtype=torch.int32)
            probs = None
            char_onehots = None
            alpha = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs,
                                                    char_onehots)
                probs_step = self.generator(hidden[0])
                hidden = (hidden[1][0], hidden[1][1])
                if probs is None:
                    probs = torch.unsqueeze(probs_step, dim=1)
                else:
                    probs = torch.cat(
                        [probs, torch.unsqueeze(
                            probs_step, dim=1)], dim=1)

                next_input = probs_step.argmax(dim=1)

                targets = next_input
        if not self.training:
            probs = F.softmax(probs, dim=2)
        return probs


class AttentionLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionLSTMCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden[1]), dim=1)

        res = torch.tanh(batch_H_proj + prev_hidden_proj)
        e = self.score(res)

        alpha = F.softmax(e, dim=1)
        alpha = torch.permute(alpha, [0, 2, 1])
        context = torch.squeeze(torch.bmm(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

        self.rnn = nn.GRUCell(input_size=input_size + num_embeddings, hidden_size=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):

        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = torch.unsqueeze(self.h2h(prev_hidden[0]), dim=1)

        res = torch.add(batch_H_proj, prev_hidden_proj)
        res = torch.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, dim=1)
        alpha = torch.permute(alpha, [0, 2, 1])
        context = torch.squeeze(torch.bmm(alpha, batch_H), dim=1)
        concat_context = torch.cat([context, char_onehots], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden[0])

        return (cur_hidden, cur_hidden), alpha