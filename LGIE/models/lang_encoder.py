import pdb
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional=False,
                 input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True, word2vec=False):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
                                 nn.ReLU()) # encoding layer after embedding
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, input_labels):
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len), must pad zero
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """
        if self.variable_lengths:
            input_lengths = (input_labels!=0).sum(1)  # Variable (batch, )

            # make ixs
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist() # list of sorted input_lengths
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist() # list of int sort_ixs, descending
            s2r = {s: r for r, s in enumerate(sort_ixs)} # O(n)
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]  # list of int recover ixs
            try:
                assert max(input_lengths_list) == input_labels.size(1)
            except:
                pdb.set_trace()

            # move to long tensor
            sort_ixs = input_labels.data.new(sort_ixs).long()  # Variable long
            recover_ixs = input_labels.data.new(recover_ixs).long()  # Variable long

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs]

        # embed
        embedded = self.embedding(input_labels)  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)            # (n, seq_len, word_vec_size)
        if self.variable_lengths:
            try:
                # TODO remove this part since there might be empty sequence
                if 0 in sorted_input_lengths_list: # if empty sequence, force assign sequence with length 1
                    pdb.set_trace()
                    sorted_input_lengths_list = np.array(sorted_input_lengths_list)
                    zero_idx = np.where(sorted_input_lengths_list == 0)[0]
                    sorted_input_lengths_list[zero_idx] = 1

                embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_input_lengths_list, batch_first=True)
            except:
                pdb.set_trace()

        # forward rnn
        output, hidden = self.rnn(embedded)

        # recover
        if self.variable_lengths:

            # embedded (batch, seq_len, word_vec_size)
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
            embedded = embedded[recover_ixs]

            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True) # (batch, max_len, hidden)
            output = output[recover_ixs]

            # recover hidden
            if self.rnn_type == 'lstm':
                hidden = hidden[0]  # we only use hidden states for the final hidden representation
            hidden = hidden[:, recover_ixs, :]  # (num_layers * num_dirs, batch, hidden_size)
            hidden = hidden.transpose(0,1).contiguous() # (batch, num_layers * num_dirs, hidden_size)
            hidden = hidden.view(hidden.size(0), -1) # (batch, num_layers * num_dirs * hidden_size)

        return output, hidden, embedded


class PhraseAttention(nn.Module):
    def __init__(self, input_dim, head_num, bias=True):
        super(PhraseAttention, self).__init__()
        # initialize pivot
        self.init_weight(head_num, input_dim, bias)
        self.reset_parameters()

    def init_weight(self, n_weight, weight_dim, bias):
        self.weight = nn.Parameter(torch.Tensor(n_weight, weight_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_weight))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, context, embedded, input_labels, head_label):
        """
        Inputs:
        - context : Variable float (batch, seq_len, input_dim)
        - embedded: Variable float (batch, seq_len, word_vec_size)
        - input_labels: Variable long (batch, seq_len)
        - head_labels: long (batch,)
        Outputs:
        - attn    : Variable float (batch, seq_len)
        - weighted_emb: Variable float (batch, word_vec_size)
        """
        weight = self.weight[head_label]
        cxt_scores = (context @ (weight.unsqueeze(2))).squeeze(2) # (batch, seq_len)

        if self.bias is not None:
            bias = self.bias[head_label]
            cxt_scores = cxt_scores + bias.unsqueeze(1)

        attn = F.softmax(cxt_scores)  # (batch, seq_len), attn.sum(1) = 1.
        # mask zeros
        is_not_zero = (input_labels!=0).float() # (batch, seq_len)
        attn = attn * is_not_zero # (batch, seq_len)
        attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1)) # (batch, seq_len)

        # compute weighted embedding
        attn3 = attn.unsqueeze(1)     # (batch, 1, seq_len)
        weighted_emb = torch.bmm(attn3, embedded) # (batch, 1, word_vec_size)
        weighted_emb = weighted_emb.squeeze(1)    # (batch, word_vec_size)

        return attn, weighted_emb


class OperatorPhraseAttention(nn.Module):
    def __init__(self, input_dim):
        super(OperatorPhraseAttention, self).__init__()
        # initialize pivot
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, context, embedded, input_labels, op_attn):
        """
        Inputs:
        - context : Variable float (batch, seq_len, input_dim)
        - embedded: Variable float (batch, seq_len, word_vec_size)
        - input_labels: Variable long (batch, seq_len)
        - op_attn: (batch, seq_len)
        - head_labels: long (batch,)
        Outputs:
        - attn    : Variable float (batch, seq_len)
        - weighted_emb: Variable float (batch, word_vec_size)
        """
        cxt_scores = self.fc(context).squeeze(2) # (batch, seq_len)
        attn = torch.softmax(cxt_scores, dim=1)  # (batch, seq_len), attn.sum(1) = 1.
        # fuse two attentions
        attn = attn * op_attn
        # mask zeros
        is_not_zero = (input_labels!=0).float() # (batch, seq_len)
        attn = attn * is_not_zero # (batch, seq_len)
        attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1)) # (batch, seq_len)

        # compute weighted embedding
        attn3 = attn.unsqueeze(1)     # (batch, 1, seq_len)
        weighted_emb = torch.bmm(attn3, embedded) # (batch, 1, word_vec_size)
        weighted_emb = weighted_emb.squeeze(1)    # (batch, word_vec_size)

        return attn, weighted_emb


class OperatorAttention(nn.Module):
    def __init__(self, input_dim, op_size, op_embedding_size):
        super(OperatorAttention, self).__init__()
        self.embedding = nn.Embedding(op_size, op_embedding_size)
        self.fc = nn.Linear(input_dim, op_embedding_size)
        self.linear_out = nn.Sequential(nn.Linear(2 * op_embedding_size, op_embedding_size),
                                        nn.BatchNorm1d(op_embedding_size),
                                        nn.ReLU())

    def forward(self, context, input_labels, op_labels):        # 这里为什么会用prior？？不是inference模式吗...
        """
        Inputs:
        - context : Variable float (batch, seq_len, input_dim)
        - input_labels: Variable long (batch, seq_len)
        - op_labels: long (batch,)
        Outputs:
        - attn    : Variable float (batch, seq_len)
        - output : Variable float (batch, word_vec_size)
        """
        op_ebd = self.embedding(op_labels)  # (batch, word_vec_size)
        context_ebd = self.fc(context)
        attn = (context_ebd @ (op_ebd.unsqueeze(2))).squeeze(2)  # (batch, seq_len)
        if len(attn.shape) < 2:
            pass
        # mask zeros
        is_not_zero = (input_labels != 0).float()  # (batch, seq_len)
        attn = attn * is_not_zero  # (batch, seq_len)
        attn = attn / attn.sum(1).view(attn.size(0), 1).expand(attn.size(0), attn.size(1)) # (batch, seq_len)

        mix = torch.bmm(context_ebd.transpose(1, 2), attn.unsqueeze(2)).squeeze(2) # (batch_size, input_dim)
        # compute weighted embedding
        comb = torch.cat((op_ebd, mix), dim=1)  # (batch_size, 2 * word_vec_size)
        output = self.linear_out(comb)  # (batch_size, word_vec_size)
        return attn, output

