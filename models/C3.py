'''
C3SINet
Copyright (c) 2019-present NAVER Corp.
MIT licenses
'''

import torch
import torch.nn as nn


class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class C3block(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        if d == 1:
            self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                                  dilation=d)
        else:
            combine_kernel = 2 * d - 1

            self.conv = nn.Sequential(
                nn.Conv2d(nIn, nIn, kernel_size=(combine_kernel, 1), stride=stride, padding=(padding - 1, 0),
                          groups=nIn, bias=False),
                nn.BatchNorm2d(nIn),
                nn.PReLU(nIn),
                nn.Conv2d(nIn, nIn, kernel_size=(1, combine_kernel), stride=stride, padding=(0, padding - 1),
                          groups=nIn, bias=False),
                nn.BatchNorm2d(nIn),
                nn.Conv2d(nIn, nIn, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False,
                          dilation=d),
                nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False))

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class Down_C3module(nn.Module):
    def __init__(self, nIn, nOut, D_rate=[2,4,8,16]):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = C3block(n, n+n1, 3, 1, D_rate[0])
        self.d2 = C3block(n, n, 3, 1, D_rate[1])
        self.d3 = C3block(n, n, 3, 1, D_rate[2])
        self.d4 = C3block(n, n, 3, 1, D_rate[3])

        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        combine = torch.cat([d1, d2, d3, d4], 1)

        # combine_in_out = input + combine
        output = self.bn(combine)
        output = self.act(output)
        return output

class C3module(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''

    def __init__(self, nIn, nOut, add=True, D_rate=[2,4,8,16]):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 4 * n

        self.c1 = C(nIn, n, 1, 1)
        self.d1 = C3block(n, n + n1, 3, 1, D_rate[0])
        self.d2 = C3block(n, n, 3, 1, D_rate[1])
        self.d3 = C3block(n, n, 3, 1, D_rate[2])
        self.d4 = C3block(n, n, 3, 1, D_rate[3])

        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d3 = self.d3(output1)
        d4 = self.d4(output1)

        combine = torch.cat([d1, d2, d3, d4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(2, stride=2))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class C3Net_Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''

    def __init__(self, classes=20, p=5, q=3, D_rate=[2,4,8,16], chnn=1.0):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        print("C3 Net Enc :  " + str(D_rate))
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)

        self.b1 = BR(16 + 3)
        self.level2_0 = Down_C3module(16 + 3, int(64*chnn),D_rate=D_rate)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(C3module(int(64*chnn), int(64*chnn),D_rate=D_rate))
        self.b2 = BR(int(128*chnn) + 3)

        self.level3_0 = Down_C3module(int(128*chnn)  + 3, int(128*chnn) ,D_rate=D_rate)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(C3module(int(128*chnn) , int(128*chnn) ,D_rate=D_rate))
        self.b3 = BR(int(256*chnn) )

        self.classifier = C(int(256*chnn) , classes, 1, 1)

    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0 = self.level1(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))

        output1_0 = self.level2_0(output0_cat)  # down-sampled
        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))

        classifier = self.classifier(output2_cat)

        return classifier


class C3Net(nn.Module):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self, classes=20, p=2, q=3, D_rate=[2,4,8,16], chnn=1.0, encoderFile=None,):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        '''
        super().__init__()
        print("C3 Net Dnc :  " + str(D_rate))


        self.encoder = C3Net_Encoder(classes, p, q, D_rate, chnn)
        # # load the encoder modules
        if encoderFile != None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            print('Encoder loaded!')
        # self.Enc_modules = []
        # for i, m in enumerate(self.encoder.children()):
        #     self.Enc_modules.append(m)


        self.level3_C = C(int(128*chnn)  + 3, classes, 1, 1)
        self.br = nn.BatchNorm2d(classes, eps=1e-03)
        self.conv = CBR(19 + classes, classes, 3, 1)

        self.up_l3 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False))
        self.combine_l2_l3 = nn.Sequential(BR(2*classes), C3module(2*classes , classes, add=False, D_rate=D_rate))

        self.up_l2 = nn.Sequential(nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False), BR(classes))

        self.classifier = nn.ConvTranspose2d(classes, classes, 2, stride=2, padding=0, output_padding=0, bias=False)

    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        output0 = self.encoder.level1(input)
        inp1 = self.encoder.sample1(input)
        inp2 = self.encoder.sample2(input)

        output0_cat = self.encoder.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.encoder.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.encoder.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.encoder.b2(torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.encoder.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.encoder.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.encoder.b3(torch.cat([output2_0, output2], 1)) # concatenate for feature map width expansion

        output2_c = self.up_l3(self.br(self.encoder.classifier(output2_cat))) #RUM

        output1_C = self.level3_C(output1_cat) # project to C-dimensional space
        comb_l2_l3 = self.up_l2(self.combine_l2_l3(torch.cat([output1_C, output2_c], 1))) #RUM

        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))

        classifier = self.classifier(concat_features)
        return classifier



def Enc_C3Net(**kwargs):
    print("C3 Net Enc :  ")
    model = C3Net_Encoder(**kwargs)
    return model

def Dnc_C3Net(**kwargs):
    print("Full C3 netDnc")
    model = C3Net(**kwargs)
    return model


