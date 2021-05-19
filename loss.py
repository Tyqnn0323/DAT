import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
criterion = nn.CrossEntropyLoss().cuda() 
def ent(output):
    return - torch.mean(output * torch.log(output + 1e-6))

#Loss

#Lcce
def loss_crossent(y_1, y_2, t):
    loss_1 = criterion(y_1, t)
    loss_2 = criterion(y_2, t)
    return loss_1, loss_2
#Ldis
def Loss_dis(out1, out2):
    out1 = F.softmax(out1,dim=0)
    out2 = F.softmax(out2,dim=0)
    assert out1.size() == out2.size()
    out1_log_softmax = F.log_softmax(out1, dim=1)
    out2_softmax = F.softmax(out2, dim=1)
    out2_log_softmax = F.log_softmax(out2, dim=1)
    out1_softmax = F.softmax(out1, dim=1)
    KL2 = F.kl_div(out1_log_softmax, out2_softmax, size_average=False)+F.kl_div(out2_log_softmax, out1_softmax, size_average=False)
    return KL2+ent(out1)+ent(out2)
