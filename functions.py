import torch
import torch.nn as nn
import pickle
import numpy as np

def load_model():
    log_dir = '../record_result/GRU/attn_local_long_GRU_modelall.pth'
    checkpoint = torch.load(log_dir)
    model = checkpoint['model']
    net = model.state_dict()

    return net

def get_emb_W():
    net = load_model()
    emb_loc_weight=net["emb_loc.weight"]
    emb_tim_weight=net["emb_tim.weight"]
    return emb_loc_weight,emb_tim_weight

def get_GRU_W():
    net = load_model()
    Wih1 = net["rnn_encoder.weight_ih_l0"]
    Whh1 = net["rnn_encoder.weight_hh_l0"]
    bih1 = net["rnn_encoder.bias_ih_l0"]
    bhh1 = net["rnn_encoder.bias_hh_l0"]
    Wih2 = net["rnn_decoder.weight_ih_l0"]
    Whh2 = net["rnn_decoder.weight_hh_l0"]
    bih2 = net["rnn_decoder.bias_ih_l0"]
    bhh2 = net["rnn_decoder.bias_hh_l0"]

    return Wih1,Whh1,bih1,bhh1,Wih2,Whh2,bih2,bhh2

def get_FCF_W():
    net = load_model()
    w = net["fc_final.weight"]
    b = net["fc_final.bias"]
    return w,b

def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r

def get_DATA2():
    filename1 = '../record_result/GRU/data_test'
    filename2 = '../record_result/GRU/test_idx'
    data_test = load_variavle(filename1)
    test_idx = load_variavle(filename2)
    return data_test , test_idx

def emb_layer(loc, tim):
    emb_loc_weight,emb_tim_weight = get_emb_W()
    emb_loc = nn.Embedding.from_pretrained(emb_loc_weight)
    emb_tim = nn.Embedding.from_pretrained(emb_tim_weight)
    ploc = emb_loc(loc)
    ptim = emb_tim(tim)
    return ploc,ptim

def TT(X):
    # 没有找到tensor数据的行向量变成列向量的方法，这里对速度应该有影响，之后再改
    b1 = X.data.cpu().numpy()
    b1 = np.array([b1])
    b1 = b1.reshape(-1, 1)
    X = torch.from_numpy(b1).cuda()
    return X

dm = 50

def p_mul1(w,x):
    #这里的abc1c2要小一些，也就是作为e的指数的要小
    a = torch.rand(x.shape[0],x.shape[1]).cuda()/dm
    b = torch.rand(w.shape[0], w.shape[1]).cuda()/dm
    c1 = torch.rand(w.shape[0], x.shape[1]).cuda()/dm
    c2 = (torch.mm(b,a) - c1).cuda()
    p1 = x - a
    p2 = w - b
    f1 = c1 + torch.mm(p2,a)
    f2 = c2 + torch.mm(b,p1) + torch.mm(p2,p1)

    return f1,f2

def p_mul1_add(w,x):
    f1,f2 = p_mul1(w,x)
    f1,f2 = add_ASS(f1,f2)

    return f1,f2

def p_mul2(w,x):
    a = torch.rand(x.shape[0],x.shape[1]).cuda()/dm
    b = torch.rand(w.shape[0], w.shape[1]).cuda()/dm
    c1 = torch.rand(w.shape[0], x.shape[1]).cuda()/dm
    c2 = (a*b - c1).cuda()
    p1 = x - a
    p2 = w - b
    f1 = c1 + a*p2
    f2 = c2 + b*p1 + p2*p1

    return f1,f2

def p_mul2_add(w,x):
    f1, f2 = p_mul2(w,x)

    f1, f2 = add_ASS(f1, f2)

    return f1,f2

def p_mul3(w,x):
    a = torch.rand(1).cuda()/ dm
    b = torch.rand(1).cuda()/ dm
    c1 = torch.rand(1).cuda()/ dm
    c2 = a*b - c1
    p1 = x - a
    p2 = w - b
    f1 = c1 + a*p2
    f2 = c2 + b*p1 + p2*p1

    return f1,f2

def p_mul3_add(w,x):
    f1, f2 =p_mul3(w,x)
    f1, f2 = add_ASS2(f1, f2)

    return f1,f2

def sigASS(ya,yb):
    a = torch.rand(1).cuda() / dm
    b = torch.rand(yb.shape[0], yb.shape[1]).cuda()/ dm
    c1 = torch.rand(yb.shape[0], yb.shape[1]).cuda()/ dm
    c2 = (a*b - c1).cuda()
    m1 = (-yb).exp() + b
    m2 = a * m1 + a*(ya).exp() - c1

    pb = m2 - c2
    pa = (-ya).exp()/a
    pb = 1/pb
    pa = 1/pa

    return pa,pb

def tanASS(ya,yb):
    pa,pb = sigASS(2*ya,2*yb)
    f1,f2 = p_mul2(pa,pb)
    f1 = 2*f1
    f2 = 2*f2 -1
    return f1,f2

def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[45m%s\033[0m"%'   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)

def add_ASS(x,y):
    f1 = torch.rand(x.shape[0], x.shape[1]).cuda()
    f2 = x + y - f1

    return f1,f2

def add_ASS2(x,y):
    f1 = torch.rand(1).cuda()
    f2 = x + y - f1

    return f1,f2