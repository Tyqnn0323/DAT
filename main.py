# -*- coding:utf-8 -*-
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10
from data.mnist import MNIST
from model import Generator
from model import Predictor
import argparse, sys
import numpy as np
import datetime
import shutil
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--result_dir', type = str, help = 'dir to output results', default = 'results/')
parser.add_argument('--noise_rate', type = float, help = 'noise rate, should be less than 1', default = 0.2)
parser.add_argument('--noise_type', type = str, help='[sym, asym]', default='sym')
parser.add_argument('--alpha', type = float, help='hyperparameter to control maximization of discrepency on classifiers', default= 0.005)
parser.add_argument('--beta', type = float, help='hyperparameter to control minimization of discrepency on generator', default= 5)
parser.add_argument('--dataset', type = str, help = 'mnist or cifar10', default = 'mnist')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--top_bn', action='store_true')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)


args = parser.parse_args()

# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr 

# load dataset
if args.dataset=='mnist':
    input_channel=1
    num_classes=10
    args.top_bn = False
    epoch_decay_start = 80
    args.n_epoch = 200
    train_dataset = MNIST(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                         )
    
    test_dataset = MNIST(root='./data/',
                               download=True,  
                               train=False, 
                               transform=transforms.ToTensor(),
                               noise_type=args.noise_type,
                               noise_rate=args.noise_rate
                        )
    
if args.dataset=='cifar10':
    input_channel=3
    num_classes=10
    args.top_bn = False
    epoch_decay_start = 80
    args.n_epoch = 200
    train_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                           )
    
    test_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transforms.ToTensor(),
                                noise_type=args.noise_type,
                                noise_rate=args.noise_rate
                          )


# Adjust learning rate and betas for Adam Optimizer (same setting with co-teaching)
alpha_plan = [learning_rate] * args.n_epoch
beta1_plan = [0.9] * args.n_epoch
for i in range(epoch_decay_start, args.n_epoch):
    alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - epoch_decay_start) * learning_rate
    beta1_plan[i] = 0.1

def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1

   
save_dir = args.result_dir +'/' +args.dataset+'/DAT/'
if not os.path.exists(save_dir):
    os.system('mkdir -p %s' % save_dir)

model_str=args.dataset+'_DAT_'+args.noise_type+'_'+str(args.noise_rate)
txtfile=save_dir+"/"+model_str+".txt"

nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
if os.path.exists(txtfile):
    os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))


# Train the Model
def train(train_loader,epoch, g, optimizer_g, h, optimizer_h, h2, optimizer_h2):
    print('Training %s...' % model_str)
    total = 0
    correct = 0
    for i, (images, labels, indexes) in enumerate(train_loader):

        if i>args.num_iter_per_epoch:
            break
      
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        feat = g(images)
        logits1 = h(feat)
        logits2 = h2(feat)
        outputs = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred1 == labels).sum()


        # Use half of the data for normal training, and the other half as sampled data for minimizing distribution divergence

        if i % 2 ==0: #minimize the ep
            loss_1, loss_2 = loss_crossent(logits1, logits2, labels)
            loss = loss_1+loss_2
            optimizer_g.zero_grad()
            optimizer_h.zero_grad()
            optimizer_h2.zero_grad()
            loss.backward()
            optimizer_g.step()
            optimizer_h.step()
            optimizer_h2.step()

        else: #calculate the divergence and minimize it on generator
        
            loss_1, loss_2 = loss_crossent(logits1, logits2, labels)
            loss = loss_1
            optimizer_h.zero_grad()
            loss.backward()
            optimizer_h.step() #labels are only used to train h (the simple h will not overfit labels)

            feat = g(images)
            logits1 = h(feat)
            logits2 = h2(feat)
            loss_dis = Loss_dis(logits1, logits2)
            loss = -args.alpha*loss_dis
            optimizer_h2.zero_grad() 
            loss.backward()
            optimizer_h2.step() #maxmize the discrepancy on h2
             
            feat = g(images)
            logits1 = h(feat)
            logits2 = h2(feat)
            loss_dis = Loss_dis(logits1, logits2)

            loss = args.beta*loss_dis
            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step() #minimize the discrepancy on g 
    acc = float(correct)/float(total)  
    return acc

# Evaluate the Model
def evaluate(test_loader, g,h):
    print ('Evaluating %s...' % model_str)
    g.eval()    # Change model to 'eval' mode.
    h.eval()
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).cuda()
        feat = g(images)
        logits = h(feat)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = float(correct)/float(total)

    return acc


def main():
    # Data Loader (Input Pipeline)
    print ('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               num_workers=4,
                                               drop_last=True,
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size, 
                                              num_workers=4,
                                              drop_last=True,
                                              shuffle=False)
    # Define models
    print ('building model...')
    g = Generator(input_channel=input_channel)
    h = Predictor(n_outputs=num_classes)
    h2 = Predictor(n_outputs=num_classes)
    g.cuda()
    h.cuda()
    h2.cuda()
    print(g.parameters,h.parameters,h2.parameters)

    optimizer_g = torch.optim.Adam(g.parameters(), lr=learning_rate)
    optimizer_h = torch.optim.Adam(h.parameters(), lr=learning_rate)
    optimizer_h2 = torch.optim.Adam(h2.parameters(), lr=learning_rate)


    with open(txtfile, "a") as myfile:
        myfile.write('epoch: train_acc test_acc\n')

    epoch=0
    train_acc=0
  
    # evaluate models with random weights
    test_acc=evaluate(test_loader,g,h)
    print('Epoch [%d/%d], Test Accuracy: %.4f' % (epoch+1, args.n_epoch, test_acc))

    # save results
    with open(txtfile, "a") as myfile:
        myfile.write(str(int(epoch)) + ': '  + str(train_acc) +' '  + str(test_acc) + "\n")

    best_ce_acc = 0
    # training
    for epoch in range(1, args.n_epoch):
        # train models
        g.train()
        h.train()
        h2.train()
        adjust_learning_rate(optimizer_g, epoch)
        adjust_learning_rate(optimizer_h, epoch)
        adjust_learning_rate(optimizer_h2, epoch)
        train_acc = train(train_loader,epoch, g, optimizer_g, h, optimizer_h, h2, optimizer_h2)
        
        # evaluate models
        test_acc = evaluate(test_loader,g,h)
        print ('Epoch [%d/%d], Training Accuracy: %.4F %%, Test Accuracy: %.4F %%' 
            %(epoch+1, args.n_epoch, train_acc*100,test_acc*100))
        with open(txtfile, "a") as myfile:
            myfile.write(str(int(epoch)) + ': '  + str(train_acc) +' '  + str(test_acc) + "\n")

if __name__=='__main__':
    main()
