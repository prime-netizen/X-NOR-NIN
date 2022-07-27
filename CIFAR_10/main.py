from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import util
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import math
from torchsummary import summary

from models import nin,nin_norelu
from torch.autograd import Variable

def save_state(model, best_acc):
    print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].copy().keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, 'models/nin.pth.tar')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
            
        # process the weights including binarization
        bin_op.binarization()
        
        # forwarding
        #data, target = Variable(data.cuda()), Variable(target.cuda())
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    for data, target in testloader:
        #data, target = Variable(data.cuda()), Variable(target.cuda())
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
                                    
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
    acc = 100. * float(correct) / len(testloader.dataset)

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(testloader.dataset),
        100. * float(correct) / len(testloader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    
    bin_op.binarization()
    for key,value in model.named_parameters():
      print(key,value)
      with open('Weights.txt', 'a') as f:
        f.write(str(key))
        f.write('\n')
        f.write('\n')
        f.write(str(value))
        f.write('\n')
        f.write('\n')
    
    return

def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.01',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    args = parser.parse_args()
    print('==> Options:',args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    #if not os.path.isfile(args.data+'/train_data'):
    if not os.path.isdir(args.data):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')
        
    to_tensor_transformer = transforms.Compose([
        transforms.ToTensor(),
        ])
    trainset = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=to_tensor_transformer)

    #trainset = data.dataset(root=args.data, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
            shuffle=True, num_workers=2)

    #testset = data.dataset(root=args.data, train=False)
    testset = torchvision.datasets.CIFAR10(args.data, train=False, download=True, transform=to_tensor_transformer)
    x=[1]
    testset = torch.utils.data.Subset(testset,x)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
            shuffle=False, num_workers=2)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    print('==> building model',args.arch,'...')
    if args.arch == 'nin':
        model = nin.NIN_train()
        model_old = nin.NIN_train()
    elif args.arch =='nin_norelu':
        model = nin_norelu.NIN_train()
        model_old = nin_norelu.NIN_train() 
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                #m.bias.data.zero_()
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model_old.load_state_dict(pretrained_model['state_dict'])
        model.load_state_dict(pretrained_model['state_dict'])
        print('Last Recorded Accuracy of pretrained model:')
        print(best_acc)


    #if not args.cpu:
    if torch.cuda.is_available():
        model.cuda()
        model_old.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':0.00001}]

    optimizer = optim.Adam(params, lr=0.10,weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)
    
    # Genration of alphas of Batch Normalization
    bn_old_list=[]
    for name,m in model_old.named_modules():
      if isinstance(m,nin.BinConv2d):
        bn_params=m.bn.running_mean-(((m.bn.running_var**0.5)*m.bn.bias)/m.bn.weight)
        bn_old_list.append(bn_params)

    i=0
    for name,m in model.named_modules():
      if isinstance(m,nin.BinConv2d):
        m.bn_params=bn_old_list[i]
        print('-' *30)
        print('Alpha starts at Bin_conv layer {}'.format(i+1))
        print('Alpha vector dimension:',m.bn_params.size(0))
        print('Max: {:.4f} | Min: {:.4f} | Mean: {:.4f} | Std: {:.4f}'.format(m.bn_params.max(),m.bn_params.min(),m.bn_params.mean(),m.bn_params.std()))
        
        with open('Alphas.txt', 'a') as f:
          f.write('Alpha starts at Bin_conv layer :')
          f.write(str(i+1))
          f.write('\n')
          f.write('The alpha values are:')
          f.write('\n')
          alp=m.bn_params.cpu().detach().numpy()
          f.write(str(alp))
          f.write('\n')
          f.write('\n')

        i+=1

    # do the evaluation if specified
    if args.evaluate:
        test()
        #bin_op.binarization()
        #print(bin_op.target_modules)
        exit(0)

    # start training
    for epoch in range(1, 320):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()
