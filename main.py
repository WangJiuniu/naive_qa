# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import math
import os
import numpy as np
import sys
sys.path.append("./")
from data_helper import Lang, prepare_data
from model import RNNReader
from arg import ARG


data1 = {"answer": "特殊教育学校", "query": "智障学校叫啥", "passage":"智障学校（也叫特殊教育学校）分为公立和私立两种。公办特殊教育学校学费全免，符合条件的智障孩子都可以报名。"}
data2 = {"answer": "2011年", "query": "微信哪一年有的","passage":"微信是腾讯公司于2011年１月２１日推出的一款手机聊天软件。２０１２年９月１７日，微信注册用户过２亿"}
data3 = {"answer": "后厨", "query": "海清和小沈阳演的电视剧","passage": "海清和小沈阳主演的电视剧　有没有妈妈看《后厨》的，感觉怎么样，我看得可是挺来劲的。里面搞笑的戏份有意思。．．．"}
data4 = {"answer": "李春波", "query": "一封家书刚开始谁唱的","passage": "唱小芳的那个　李春波　［如果我的回答对您有帮助　请点击＂好评＂支持下　谢谢］"}
data5 = {"answer": "李春波", "query": "小芳谁唱的","passage": "唱小芳的那个人叫 李春波，谢谢"}


def get_batch(data_all, batch_size=1):
    q_idx, q_mask, p_idx, p_mask, s_pos, e_pos = data_all
    batches = math.floor(len(q_idx) * 1.0 /batch_size)
    for i in range(batches):
        start = batch_size * i
        end = batch_size * (i+1)
        batch_data = [q_idx[start:end], q_mask[start:end], p_idx[start:end],
                      p_mask[start:end], s_pos[start:end], e_pos[start:end]]
        yield batch_data

def train(model, optimer, train_data_all, test_data_all, test_data,epochs=100):
    for epoch in range(epochs):
        batches = get_batch(train_data_all, model.arg.batch_size)
        for batch in batches:
            # 准备batch中的数据
            model.train()
            q_idx, q_mask, p_idx, p_mask, real_s, real_e = batch
            q_idx = Variable(torch.LongTensor(q_idx))
            q_mask = Variable(torch.LongTensor(q_mask).byte())
            p_idx = Variable(torch.LongTensor(p_idx))
            p_mask = Variable(torch.LongTensor(p_mask).byte())
            inputs = (q_idx, q_mask, p_idx, p_mask)
            # 送入模型进行预测
            score_s, score_e = model(*inputs)
            # print("pre_s:",pre_s.size())
            # bs, p_len = pre_s.size()
            target_s = Variable(torch.LongTensor(real_s))
            target_e = Variable(torch.LongTensor(real_e))
            # 计算loss并更新
            # print("score_s:", score_s.size())
            # print("score_s:", score_s)
            # print("target_s:",target_s.size(), target_s)
            loss = F.cross_entropy(score_s, target_s) + F.cross_entropy(score_e, target_e)
            print('epoch:{}\tloss:{}\t'.format(epoch, loss.data[0]))
            optimer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm(model.parameters(),
                                          model.arg.grad_clipping)
            optimer.step()
        predict(model, test_data_all, test_data)


def predict(model, test_data_all, test_data):
    q_idx, q_mask, p_idx, p_mask, real_s, real_e = test_data_all
    q_idx = Variable(torch.LongTensor(q_idx))
    q_mask = Variable(torch.LongTensor(q_mask).byte())
    p_idx = Variable(torch.LongTensor(p_idx))
    p_mask = Variable(torch.LongTensor(p_mask).byte())
    inputs = (q_idx, q_mask, p_idx, p_mask)
    score_s, score_e = model(*inputs)
    s = np.argmax(score_s.data.numpy())
    e = np.argmax(score_e.data.numpy())
    if s < e and e < len(test_data[0]["passage"]):
        try :
            print(test_data[0]["passage"][s:e])
        except:
            print('error!')
    else:
        print("Requirement: s < e")



def main():
    train_data = [data1, data2, data3, data4]
    test_data = [data5]
    lang = Lang()
    arg = ARG()
    lang.insert_data(train_data)
    train_data_all = prepare_data(train_data, lang.char2idx, arg)
    test_data_all = prepare_data(test_data, lang.char2idx, arg)
    # print(lang.char2idx)

    # prepare for train
    model = RNNReader(arg, len(lang.char2idx))
    optimer = optim.SGD(model.parameters(), lr=1e-4)
    # train model
    model_file = 'model_1.pkl'
    if os.path.exists(model_file):
        model = torch.load(model_file)
    else:
        train(model, optimer, train_data_all, test_data_all, test_data)
        torch.save(model, model_file)
    predict(model, test_data_all, test_data)


if __name__ == '__main__':
    main()







