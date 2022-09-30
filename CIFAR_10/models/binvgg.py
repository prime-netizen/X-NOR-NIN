#Architecture NIN. Modifications: Relu,bias units removed from bin conv layers
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        
        """
        Saving the binarized weights
        """
        bin_input=input.detach().cpu().numpy()
        file=open('Binarized Inputs.txt','a')
        file.write('The Binarized Inputs for the layer are:')
        file.write('\n')
        file.write('\n')           
        for i in bin_input:
            for j in i:
                np.savetxt(file,j)
        file.close()
        
        
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

    
class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0, save_info=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'Bin_Conv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout
        self.save_info= save_info
        
        self.bn_params = torch.zeros(input_channels)
        self.dist_margin = torch.zeros(output_channels)

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        self.bn.weight.data = self.bn.weight.data.zero_().add(1.0)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        #self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x_value = x.clone()
        x = self.bn(x)
        ## Compact Batch Normalization
       # x_bn=torch.zeros_like(x)
       # if torch.cuda.is_available():
         #   x_bn=x_bn.cuda()
        #for i in range(self.bn_params.size(0)):
         #   x_bn[:,i,:,:]=self.bn_params[i]
        #x=x-x_bn
        
        x, mean = BinActive.apply(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        
        ## Add variations
        #var_x=torch.ones_like(x)
        #if torch.cuda.is_available():
        #    var_x=var_x.cuda()
       # for i in range(self.dist_margin.size(0)):
          #  if torch.cuda.is_available():
             #   var_x[:,i,:,:]=var_x[:,i,:,:]*self.dist_margin[i].cuda()
           # else:
              #  var_x[:,i,:,:]=var_x[:,i,:,:]*self.dist_margin[i]
       # x=x+var_x
                       
        #x = BinOp.binarization(x)
        if self.save_info:
            save_variable(x_value,self.bn.weight.data,self.bn.bias.data,self.conv.weight.data,self.conv.bias.data, x )
        #x = self.relu(x)
        return x

    
    

cfg = {
    'VGG11': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class Bin_VGG_train(nn.Module):
    def __init__(self, vgg_name):
        super(Bin_VGG_train, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                m.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = OrderedDict([
            ('conv0', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True))
            ])
        in_channels = 64
        cnt = 1
        for x in cfg:
            if x == 'M':
                layers['pool'+str(cnt)] = nn.MaxPool2d(kernel_size=2, stride=2)
                cnt += 1
            else:
                layers['conv'+str(cnt)] = BinConv2d(input_channels=in_channels, output_channels=x, kernel_size=3, stride=1, padding=1)
                cnt += 1
                layers['bn'+str(cnt)] = nn.BatchNorm2d(x)
                cnt += 1
                layers['relu'+str(cnt)] = nn.ReLU(inplace=True)
                cnt += 1
                in_channels = x
        layers['pool'+str(cnt)] = nn.AvgPool2d(kernel_size=1, stride=1)
        return nn.Sequential(layers)


class Bin_VGG_test(nn.Module):
    def __init__(self, vgg_name):
        super(Bin_VGG_test, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = OrderedDict([
            ('conv0', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True))
            ])
        in_channels = 64
        cnt = 1
        for x in cfg:
            if x == 'M':
                layers['pool'+str(cnt)] = nn.MaxPool2d(kernel_size=2, stride=2)
                cnt += 1
            else:
                layers['conv'+str(cnt)] = BinConv2d(input_channels=in_channels, output_channels=x, kernel_size=3, stride=1, padding=1)
                cnt += 1
                layers['bn'+str(cnt)] = nn.BatchNorm2d(x)
                cnt += 1
                layers['relu'+str(cnt)] = nn.ReLU(inplace=True)
                cnt += 1
                in_channels = x
        layers['pool'+str(cnt)] = nn.AvgPool2d(kernel_size=1, stride=1)
        return nn.Sequential(layers)



    
def save_variable(x,bn_weights,bn_bias,conv_weights,conv_bias,output):
    """
    Inputs
    """
    inputs = x.detach().cpu().numpy()
    file = open('inputs.txt','a')    
    for i in inputs:
        for j in i:           
            np.savetxt(file,j) 
    file.close()


    """
    Batch Normalization Weights
    """
    bn_weights = bn_weights.detach().cpu().numpy()
    file = open('bn_weights.txt','a')    
    np.savetxt(file,bn_weights)
    file.close()

    
    """
    Batch Normalization Bias
    """

    bn_bias = bn_bias.detach().cpu().numpy()
    file = open('bn_bias.txt','a')    
    np.savetxt(file,bn_bias)
    file.close()

    
    """
    Convolution Weights
    """  
    conv_weights = conv_weights.detach().cpu().numpy()
    file = open('conv_weights.txt','a')    
    for i in conv_weights:
        for j in i:           
            np.savetxt(file,j)  
    file.close()
    

    
    """
    Convolution Bias
    """
    conv_bias = conv_bias.detach().cpu().numpy()
    file = open('conv_bias.txt','a')
    np.savetxt(file,conv_bias)
    file.close()
    

    """
    Output Activation
    """
    output = output.detach().cpu().numpy()
    file = open('pre-activation_outputs.txt','a')    
    for i in output:
        for j in i:          
            np.savetxt(file,j)
    file.close()
