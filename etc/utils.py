from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator
import numpy as np
count_ops = 0
count_params = 0


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)
    ### ops_conv
    if type_name in ['Conv2d' , 'ConvTranspose2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.dilation[0]*(layer.kernel_size[0] -1) - 1) / layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.dilation[1]*(layer.kernel_size[1] -1) - 1) / layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)
        # print(str(layer) + " :" + str(delta_params))

       ### ops_nonlinearity
    elif type_name in ['ReLU', 'PReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    elif type_name in ['BatchNorm2d']:
        delta_params = get_layer_param(layer)
        delta_ops = delta_params

    ### unknown layer type
    # else:
    #     raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, H, W))

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params


#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
#============================================

class MyLRScheduler(object):
    '''
    CLass that defines cyclic learning rate that decays the learning rate linearly till the end of cycle and then restarts
    at the maximum value.
    '''
    def __init__(self, initial=0.1, cycle_len=5, ep_cycle=50, ep_max=100):
        super(MyLRScheduler, self).__init__()

        self.min_lr = initial# minimum learning rate
        self.m = cycle_len
        self.ep_cycle = ep_cycle
        self.ep_max = ep_max
        self.poly_start = initial
        self.step = initial/ self.ep_cycle
        print('Using Cyclic LR Scheduler with warm restarts and poly step'
              + str(self.step))

    def get_lr(self, epoch):
        if epoch==0:
            current_lr = self.min_lr
        elif 0< epoch and epoch <= self.ep_cycle:
            counter = (epoch-1) % self.m
            current_lr = round((self.min_lr * self.m) - (counter * self.min_lr), 5)
        else:

            current_lr = round(self.poly_start - (epoch-self.ep_cycle )*self.step, 8)

            # current_lr = round(self.poly_start * (1 - (epoch-self.ep_cycle) / (self.ep_max-self.ep_cycle)) ** 0.9, 8)

        return current_lr


class WarmupPoly(object):
    '''
    CLass that defines cyclic learning rate that decays the learning rate linearly till the end of cycle and then restarts
    at the maximum value.
    '''
    def __init__(self, init_lr, total_ep, warmup_ratio=0.05, poly_pow = 0.98):
        super(WarmupPoly, self).__init__()
        self.init_lr = init_lr
        self.total_ep = total_ep
        self.warmup_ep = int(warmup_ratio*total_ep)
        print("warup unitl " + str(self.warmup_ep))
        self.poly_pow = poly_pow

    def get_lr(self, epoch):
        #
        if epoch < self.warmup_ep:
            curr_lr =  self.init_lr*pow((((epoch+1) / self.warmup_ep)), self.poly_pow)

        else:
            curr_lr = self.init_lr*pow((1 - ((epoch- self.warmup_ep)  / (self.total_ep-self.warmup_ep))), self.poly_pow)

        return curr_lr

def colormap_cityscapes(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)
    cmap[0, :] = np.array([128, 64, 128])
    cmap[1, :] = np.array([244, 35, 232])
    cmap[2, :] = np.array([70, 70, 70])
    cmap[3, :] = np.array([102, 102, 156])
    cmap[4, :] = np.array([190, 153, 153])
    cmap[5, :] = np.array([153, 153, 153])

    cmap[6, :] = np.array([250, 170, 30])
    cmap[7, :] = np.array([220, 220, 0])
    cmap[8, :] = np.array([107, 142, 35])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])

    cmap[11, :] = np.array([220, 20, 60])
    cmap[12, :] = np.array([255, 0, 0])
    cmap[13, :] = np.array([0, 0, 142])
    cmap[14, :] = np.array([0, 0, 70])
    cmap[15, :] = np.array([0, 60, 100])

    cmap[16, :] = np.array([0, 80, 100])
    cmap[17, :] = np.array([0, 0, 230])
    cmap[18, :] = np.array([119, 11, 32])
    cmap[19, :] = np.array([0, 0, 0])

    return cmap

class Colorize:

    def __init__(self, n=22):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        #print(size)
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(255)
        #color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            #mask = gray_image == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    max_epochs = 300
    # lrSched = MyLRScheduler(initial=0.007, cycle_len=10, ep_cycle=150, ep_max=300)#MyLRScheduler(0.1, 5, [51, 101, 131, 161, 191, 221, 251, 281, 311, 341, 371])
    lrSched = WarmupPoly(5e-4, max_epochs , poly_pow=0.95)

    x=[]
    y=[]
    for i in range(max_epochs):
        x.append(i)
        y.append(lrSched.get_lr(i))
        print(lrSched.get_lr(i))
    plt.plot(x,y)

    plt.show()

