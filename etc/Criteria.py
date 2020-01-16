import torch.nn as nn
import torch.nn.functional as F
import torch


class CrossEntropyLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''
    def __init__(self, weight=None, ignore_id = None):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        '''

        super().__init__()
        if ignore_id != None:
            self.loss = nn.NLLLoss(weight, ignore_index=ignore_id)
        else:
            self.loss = nn.NLLLoss(weight)



    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)


class LabelingSmooth_CrossEntropyLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''
    def __init__(self, nCls, smooth=0.1):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        '''

        super().__init__()
        self.smooth = smooth
        self.nCls = nCls

    def forward(self, outputs, targets):
        pixel_size = outputs.size(0)*outputs.size(2)*outputs.size(3)
        smooth_target= torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1)
        smooth_target = smooth_target * (1 - self.smooth) + (1 - smooth_target) * self.smooth / (self.nCls - 1)
        loss = - (smooth_target * F.log_softmax(outputs, 1)).sum() / pixel_size
        return loss