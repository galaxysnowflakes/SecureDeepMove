# encoding=UTF-8
#所有能加的地方都加了

import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import numpy as np
import random
from decimal import *
import math
import time

torch.set_printoptions(precision=10)

from train import get_acc, generate_queue , generate_input_history, \
    generate_input_long_history, generate_input_long_history2

from functions import get_GRU_W,get_FCF_W,add_ASS,add_ASS2\
    ,get_DATA2,emb_layer,TT,p_mul1_add,p_mul2_add,p_mul3_add,sigASS,tanASS,process_bar

def GRU_ASS(Wih1, Whh1, bih1, bhh1,X1,lenth):
    X1_output1 = []
    X1_output2 = []
    #out = []

    Whr1 = Whh1[:lenth, :]
    Whz1 = Whh1[lenth:2 * lenth, :]
    Whn1 = Whh1[2 * lenth:3 * lenth, :]

    ma1, ma2 = p_mul1_add(Wih1, X1)
    ma2 = ma2 + bih1
    ma2[:2 * lenth, :] = ma2[:2 * lenth, :] + bhh1[:2 * lenth]

    ma1, ma2 = add_ASS(ma1, ma2)

    r1 = ma1[:lenth, :]
    r2 = ma2[:lenth, :]
    z1 = ma1[lenth:2 * lenth, :]
    z2 = ma2[lenth:2 * lenth, :]
    n1 = ma1[2 * lenth:3 * lenth, :]
    n2 = ma2[2 * lenth:3 * lenth, :]

    # 计算第一个rt
    mid_r1, mid_r2 = sigASS(r1[:, :1], r2[:, :1])
    rt1, rt2 = p_mul2_add(mid_r1, mid_r2)
    # ans1 = rt1 + rt2
    # ans2 = sig(torch.mm(Wih1,X1)[:lenth, :] + bih1[:lenth, :] + bhh1[:lenth, :])

    # 计算第一个zt
    mid_z1, mid_z2 = sigASS(z1[:, :1], z2[:, :1])
    zt1, zt2 = p_mul2_add(mid_z1, mid_z2)

    # 计算第一个nt
    bhn = bhh1[2 * lenth:3 * lenth, :1]
    f1, f2 = p_mul2_add(rt1, bhn)
    mid_n1 = n1[:, :1] + f1
    mid_n2 = n2[:, :1] + f2 + rt2 * bhn
    mid_n1,mid_n2 = add_ASS(mid_n1,mid_n2)
    nt1, nt2 = tanASS(mid_n1, mid_n2)

    # 计算第一个ht
    f1, f2 = p_mul2_add(0.5 - zt1, nt2)
    g1, g2 = p_mul2_add(0.5 - zt2, nt1)
    ht1 = (0.5 - zt1) * nt1 + f1 + g1
    ht2 = (0.5 - zt2) * nt2 + f2 + g2
    ht1,ht2 = add_ASS(ht1,ht2)
    #print(ht1.shape)
    X1_output1.append(ht1)
    X1_output2.append(ht2)
    #out.append((ht1+ht2))

    for i in range(1, ma1.shape[1]):  #

        # 计算下一轮rt
        f1, f2 = p_mul1_add(Whr1, ht1)
        mid_r1 = r1[:, i:i + 1] + f1
        mid_r2 = r2[:, i:i + 1] + f2 + torch.mm(Whr1, ht2)
        mid_r1, mid_r2 = add_ASS(mid_r1, mid_r2)

        rt1, rt2 = sigASS(mid_r1, mid_r2)
        rt1, rt2 = p_mul2_add(rt1, rt2)
        # ans1 = rt1 + rt2
        # ans2 = sig(r1[:,1:2]+r2[:,1:2]+torch.mm(Whr1,ht1+ht2))
        #print("r:", rt1.shape, rt2.shape)

        # 计算下一轮zt
        f1, f2 = p_mul1_add(Whz1, ht1)
        mid_z1 = z1[:, i:i + 1] + f1
        mid_z2 = z2[:, i:i + 1] + f2 + torch.mm(Whz1, ht2)
        zt1, zt2 = sigASS(mid_z1, mid_z2)
        zt1, zt2 = p_mul2_add(zt1, zt2)
        #print("z:", zt1.shape, zt2.shape)
        # ans1 = zt1+zt2
        # ans2 = sig(z1[:,1:2] + z2[:,1:2] +torch.mm(Whz1,ht2+ht1))

        # 计算下一轮nt
        f1, f2 = p_mul1_add(Whn1, ht1)
        f2 = f2 + torch.mm(Whn1, ht2) + bhh1[2 * lenth:3 * lenth]
        f1, f2 = add_ASS(f1,f2)
        #print("f:", f1.shape, f2.shape)
        q1, q2 = p_mul2_add(rt1, f2)
        q3, q4 = p_mul2_add(rt2, f1)
        #print("q:", q1.shape, q2.shape)
        mid_nt1 = rt1 * f1 + q1 + q3 + n1[:, i:i + 1]
        mid_nt2 = rt2 * f2 + q2 + q4 + n2[:, i:i + 1]
        mid_nt1, mid_nt2 = add_ASS(mid_nt1, mid_nt2)
        nt1, nt2 = tanASS(mid_nt1, mid_nt2)
        # ans1 = nt1+nt2
        # ans2 = tan(n1[:,1:2] + n2[:,1:2] +(rt1+rt2)*(torch.mm(Whn1,ht1+ht2)+bhh1[2*lenth:3*lenth]))
        #print("n:", nt1.shape, nt2.shape)

        # 计算下一轮ht
        f1, f2 = p_mul2_add(0.5 - zt1, nt2)
        f3, f4 = p_mul2_add(0.5 - zt2, nt1)
        g1, g2 = p_mul2_add(zt1, ht2)
        g3, g4 = p_mul2_add(zt2, ht1)
        # ans1 = (1 - zt1 - zt2) * (nt1 + nt2) + (zt1 + zt2) * (ht1 + ht2)
        ht1 = zt1 * ht1 + f1 + f3 + g1 + g3 + (0.5 - zt1) * nt1
        ht2 = zt2 * ht2 + f2 + f4 + g2 + g4 + (0.5 - zt2) * nt2
        ht1, ht2 = add_ASS(ht1, ht2)
        # ans2 = ht1 + ht2
        #print("h:", ht1.shape, ht2.shape)

        X1_output1.append(ht1)
        X1_output2.append(ht2)
        #out.append((ht1 + ht2))

    return X1_output1,X1_output2#,out

#每句后边标注了#的是之后需要修改的
def runASS(data,run_idx):
    run_queue = generate_queue(run_idx, 'normal', 'test')
    queue_len = len(run_queue)
    users_acc = {}
    for c in range(queue_len): #queue_len  1334——就会报错了
        end_str = '100%'
        process_bar(c / queue_len, start_str='', end_str=end_str, total_length=15)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        u, i = run_queue.popleft()
        if u not in users_acc:
            users_acc[u] = [0, 0, 0, 0]
        loc = data[u][i]['loc'].cuda()
        tim = data[u][i]['tim'].cuda()
        target = data[u][i]['target'].cuda()

        target_len = target.data.size()[0]

        #embedding层
        loc_emb,tim_emb = emb_layer(loc,tim)
        x = torch.cat((loc_emb, tim_emb), 2)   #x的范围在±3之内

        #GRU层
        print("**************GRU**************")
        Wih1, Whh1, bih1, bhh1, Wih2, Whh2, bih2, bhh2 = get_GRU_W()
        bih1 = TT(bih1)
        bhh1 = TT(bhh1)
        bih2 = TT(bih2)
        bhh2 = TT(bhh2)

        lenth = int(bih1.shape[0] / 3)

        #这里的h1 h2对应源代码里的hidden_history  hidden_state
        #计算h1
        X1 = x[:-target_len].squeeze(1).t()
        X1_output1, X1_output2 = GRU_ASS(Wih1, Whh1, bih1, bhh1,X1,lenth)
        #这里的hh1等同于下面计算的h1
        #h1 = GRU_layer(Wih1, Whh1, bih1, bhh1, x[:-target_len])


        # 计算h2
        X2 = x[-target_len:].squeeze(1).t()
        X2_output1, X2_output2 = GRU_ASS(Wih2, Whh2, bih2, bhh2, X2, lenth)
        # 这里的hh2等同于下面计算的h2
        #h2 = GRU_layer(Wih2, Whh2, bih2, bhh2, x[-target_len:])

        #把list类转成tensor类
        #维度是48 * 300
        # X1_1+X1_2 = 原h1
        # X2_1+X2_2 = 原h2
        X1_1 = torch.Tensor([item.cpu().detach().numpy() for item in X1_output1]).cuda()
        X1_2 = torch.Tensor([item.cpu().detach().numpy() for item in X1_output2]).cuda()
        X2_1 = torch.Tensor([item.cpu().detach().numpy() for item in X2_output1]).cuda()
        X2_2 = torch.Tensor([item.cpu().detach().numpy() for item in X2_output2]).cuda()
        X1_1 = X1_1.squeeze(2)
        X1_2 = X1_2.squeeze(2)
        X2_1 = X2_1.squeeze(2)
        X2_2 = X2_2.squeeze(2)
        X1_1 ,X1_2 = add_ASS(X1_1 ,X1_2)
        X2_1, X2_2 = add_ASS(X2_1, X2_2)



        #attn层
        seq_len = X1_1.size()[0]
        state_len = X2_1.size()[0]
        attn_energies1 = Variable(torch.zeros(state_len, seq_len)).cuda()
        attn_energies2 = Variable(torch.zeros(state_len, seq_len)).cuda()
        #aeans = Variable(torch.zeros(state_len, seq_len)).cuda()

        for i in range(state_len):
            for j in range(seq_len):
                f1,f2 = p_mul1_add(X1_1[j:j+1,:], X2_2[i:i+1,:].t())
                f3,f4 = p_mul1_add(X1_2[j:j+1,:], X2_1[i:i+1,:].t())
                attn_energies1[i, j] = f1 + f3 + X1_1[j].dot(X2_1[i])
                attn_energies2[i, j] = f2 + f4 + X1_2[j].dot(X2_2[i])
                #aeans[i, j] = (attn_energies1[i, j] + attn_energies2[i, j])


        attn_energies1, attn_energies2 = add_ASS(attn_energies1, attn_energies2)
        random_a = random.randint(1, 100)
        random_a = random_a / 100
        attn_energies1 = attn_energies1.exp()
        attn_energies1 = attn_energies1 * random_a
        attn_energies2 = attn_energies2.exp()

        sm1 = Variable(torch.zeros(state_len, seq_len)).cuda()
        sm2 = Variable(torch.zeros(state_len, seq_len)).cuda()
        c1 = Variable(torch.zeros(state_len).cuda())
        c2 = Variable(torch.zeros(state_len).cuda())

        #print(attn_energies1[i, j].shape)

        for i in range(state_len):
            c1[i] = 0
            c2[i] = 0
            for j in range(seq_len):
                f1, f2 = p_mul3_add(attn_energies1[i, j],attn_energies2[i, j])
                c1[i] = c1[i] + f1
                c2[i] = c2[i] + f2
                sm1[i,j] = f1
                sm2[i,j] = f2
            for j in range(seq_len):
                sm1[i,j] = sm1[i,j] / (c1[i] + c2[i])
                sm2[i,j] = sm2[i,j] / (c1[i] + c2[i])

        #这里的sm1和sm2就开始有误差了
        #attn_weights1 = sm1 + sm2
        #print(attn_weights1.shape)
        sm1, sm2 = add_ASS(sm1, sm2)
        sm1 = sm1.unsqueeze(0)
        sm2 = sm2.unsqueeze(0)


        f1,f2 = p_mul1_add(sm1[0],X1_2.unsqueeze(0)[0])
        g1,g2 = p_mul1_add(sm2[0],X1_1.unsqueeze(0)[0])

        context1 = torch.mm(sm1[0],X1_1.unsqueeze(0)[0]) + f1 + g1
        context2 = torch.mm(sm2[0],X1_2.unsqueeze(0)[0]) + f2 + g2

        context1,context2 = add_ASS(context1,context2)

        out1 = torch.cat((X2_1, context1), 1)
        out2 = torch.cat((X2_2, context2), 1)

        # fc_final层
        #误差不小
        fc_finalw, fc_finalb = get_FCF_W()
        f1 ,f2 = p_mul1_add(out1,fc_finalw.t())
        y1 = f1
        y2 = torch.mm(out2,fc_finalw.t()) + fc_finalb + f2

        scores = nn.functional.log_softmax(y1+y2)


        if scores.data.size()[0] > target.data.size()[0]:
            scores = scores[-target.data.size()[0]:]
        #loss = criterion(scores, target)  # 就是计算损失的常见操作  #这里的意思是在比较损失，其实就相当于是计算准确率？？

        users_acc[u][0] += len(target)
        acc = get_acc(target, scores)
        users_acc[u][1] += acc[0]  # top10
        users_acc[u][2] += acc[1]  # top5
        users_acc[u][3] += acc[2]  # top1

        # total_loss.append(loss.data.cpu().numpy())  # 这里去掉了[0]

        # avg_loss = np.mean(total_loss, dtype=np.float64)

    users_rnn_acc10 = {}
    for u in users_acc:
        tmp_acc = users_acc[u][1] / users_acc[u][0]
        users_rnn_acc10[u] = tmp_acc.tolist()[0]
    avg_acc10 = np.mean([users_rnn_acc10[x] for x in users_rnn_acc10])

    users_rnn_acc5 = {}
    for u in users_acc:
        tmp_acc = users_acc[u][2] / users_acc[u][0]
        users_rnn_acc5[u] = tmp_acc.tolist()[0]
    avg_acc5 = np.mean([users_rnn_acc5[x] for x in users_rnn_acc5])

    users_rnn_acc = {}
    for u in users_acc:
        tmp_acc = users_acc[u][3] / users_acc[u][0]
        users_rnn_acc[u] = tmp_acc.tolist()[0]
    avg_acc = np.mean([users_rnn_acc[x] for x in users_rnn_acc])

    return queue_len, avg_acc, avg_acc5, avg_acc10  # , users_rnn_acc  #avg_loss,

#运行测试
def begin(dm2):
    global dm
    dm = dm2
    with torch.no_grad():
        print("try7")
        data_test, test_idx = get_DATA2()
        st = time.time()
        l, acc1, acc5, acc10 = runASS(data_test, test_idx)
        tt = time.time() - st
        print("top1:", acc1)
        print("top5:", acc5)
        print("top10:", acc10)
        return acc1,acc5,acc10,tt

#st = time.time()
#begin(1)
#print('total time cost:{}'.format(time.time() - st))










