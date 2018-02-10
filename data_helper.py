# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


# make char2idx
class Lang():
    def __init__(self):
        self.char2idx = {"#PAD#": 0, "#OOV#": 1}
        self.idx2char = {}

    def insert(self, sentence):
        for char in sentence:
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx)

    def insert_data(self, train_data):
        for data in train_data:
            self.insert(data["query"])
            self.insert(data["passage"])


# prepare data
def prepare_sequence(seq, to_ix, max_length):
    length = len(seq)
    if length < max_length:
        idxs = [(to_ix[w] if w in to_ix else to_ix["#OOV#"]) for w in seq] \
               + [to_ix["#PAD#"] for _ in range(max_length - length)]
        mask = [0] * length + [1] * (max_length - length)
    else:
        idxs = [to_ix[w] for w in seq[:max_length]]
        mask = [0] * max_length
    return idxs, mask

def get_position(passage, answer):
    splited = passage.split(answer)
    b_pos = len(splited[0]) - 1
    e_pos = b_pos + len(answer)
    return b_pos, e_pos


def prepare_data(raw_data, char2idx, arg):
    q_idx, q_mask, p_idx, p_mask, b_pos, e_pos = [], [], [], [], [], []
    for data in raw_data:
        sub_q_idx, sub_q_mask = prepare_sequence(data["query"], char2idx, arg.Q_MAX_LEN)
        sub_p_idx, sub_p_mask = prepare_sequence(data["passage"], char2idx, arg.P_MAX_LEN)
        sub_b_pos, sub_e_pos = get_position(data["passage"], data["answer"])
        q_idx.append(sub_q_idx)
        q_mask.append(sub_q_mask)
        p_idx.append(sub_p_idx)
        p_mask.append(sub_p_mask)
        b_pos.append(sub_b_pos)
        e_pos.append(sub_e_pos)
    return q_idx, q_mask, p_idx, p_mask, b_pos, e_pos








