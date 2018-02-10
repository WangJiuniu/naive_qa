# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from learn_torch.S07_naive_qa import layers
# reader model
class RNNReader(nn.Module):
    def __init__(self, arg, vocab_size, normalize=True):
        super(RNNReader,self).__init__()
        self.arg = arg
        self.P_MAX_LEN = arg.P_MAX_LEN
        self.Q_MAX_LEN = arg.Q_MAX_LEN
        self.embedding = nn.Embedding(vocab_size, arg.embedding_dim)

        self.q_rnn = layers.StackedBRNN(
            input_size=arg.embedding_dim,
            hidden_size=arg.hidden_dim,
            num_layers = arg.q_rnn_layers
        )

        p_input_size = arg.embedding_dim + arg.q_rnn_layers * arg.hidden_dim
        # print("p_input_size:",p_input_size)
        self.p_rnn = layers.StackedBRNN(
            input_size=p_input_size,
            hidden_size=arg.hidden_dim,
            num_layers=arg.p_rnn_layers
        )

        doc_hidden_size = 2 * arg.hidden_dim
        question_hidden_size = 2 * arg.hidden_dim
        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def forward(self,  q, q_mask, p, p_mask,):
        p_embed = self.embedding(p) #.permute(1,0,2)
        q_embed = self.embedding(q) #.permute(1,0,2)
        # print('p_embed:', p_embed.size())
        # print('q_embed:', q_embed.size())


        q_hiddens = self.q_rnn(q_embed, q_mask)
        # print('q_hiddens:', q_hiddens.size())
        q_hidden = q_hiddens[:,-1,:]
        # print("q_hidden:",q_hidden.size())
        q_represent = [q_hidden.resize(self.arg.batch_size, 1,  2 * self.arg.hidden_dim)] * self.arg.P_MAX_LEN
        q_represent = torch.cat(q_represent, 1)
        # print('p_embed:', p_embed.size())
        # print('q_represent:',q_represent.size())
        p_rnn_in = torch.cat((p_embed, q_represent), 2)

        p_hiddens = self.p_rnn(p_rnn_in, p_mask)
        # print('p_hiddens:', p_hiddens.size())


        start_scores = self.start_attn(p_hiddens, q_hidden, p_mask)
        end_scores = self.end_attn(p_hiddens, q_hidden, p_mask)
        return start_scores, end_scores