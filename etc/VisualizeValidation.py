'''
C3SINet
Copyright (c) 2019-present NAVER Corp.
MIT license
'''
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import cv2
import numpy as np
import torch
from torch.autograd import Variable
import glob
import json
from PIL import Image
from torchvision.transforms import functional as F

import time
from argparse import ArgumentParser
import models
from etc.utils import *

parser = ArgumentParser()
# # parser.add_argument('--model', default="gDAsymNL_ESP_1", help='Model name')
parser.add_argument('-c', '--config', type=str, default='', help='JSON file for configuration')
parser.add_argument('--data_dir', default="../../DATA/cityscape/leftImg8bit/Val", help='Data directory')
parser.add_argument('--img_extn', default="png", help='RGB Image format')

parser.add_argument('--modelType', type=int, default=2, help='1=Not do, 2=Do something')
parser.add_argument('--savedir', default='./Valserver', help='directory to save the results')
parser.add_argument('--gpu', default=True, type=bool, help='Run on CPU or GPU. If TRUE, then GPU.')
parser.add_argument('--cityFormat', default=True, type=bool, help='If you want to convert to cityscape '
                                                                  'original label ids')
parser.add_argument('--colored', default=True, type=bool, help='If you want to visualize the '
                                                               'segmentation masks in color')
parser.add_argument('--overlay', default=True, type=bool, help='If you want to visualize the '
                                                                'segmentation masks overlayed on top of RGB image')

args = parser.parse_args()

def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p
        # print(total_paramters)

    return total_paramters


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
Val_folder = ["frankfurt", "lindau", "munster"]
pallete = [[128, 64, 128],
           [244, 35, 232],
           [70, 70, 70],
           [102, 102, 156],
           [190, 153, 153],

           [153, 153, 153],
           [250, 170, 30],
           [220, 220, 0],
           [107, 142, 35],
           [152, 251, 152],

           [70, 130, 180],
           [220, 20, 60],
           [255, 0, 0],
           [0, 0, 142],
           [0, 0, 70],

           [0, 60, 100],
           [0, 80, 100],
           [0, 0, 230],
           [119, 11, 32],
           [0, 0, 0]]


def relabel(img):
    '''
    This function relabels the predicted labels so that cityscape dataset can process
    :param img:
    :return:
    '''
    img[img == 19] = 255
    img[img == 18] = 33
    img[img == 17] = 32
    img[img == 16] = 31
    img[img == 15] = 28
    img[img == 14] = 27
    img[img == 13] = 26
    img[img == 12] = 25
    img[img == 11] = 24
    img[img == 10] = 23
    img[img == 9] = 22
    img[img == 8] = 21
    img[img == 7] = 20
    img[img == 6] = 19
    img[img == 5] = 17
    img[img == 4] = 13
    img[img == 3] = 12
    img[img == 2] = 11
    img[img == 1] = 8
    img[img == 0] = 7
    img[img == 255] = 0
    return img


def evaluateModel(args, model, image_list, folder_name, inputsize):
    # gloabl mean and std values
    mean = [72.3923111, 82.90893555, 73.15840149]
    std = [45.3192215, 46.15289307, 44.91483307]

    if not os.path.exists(os.path.join(args.savedir,'Val',folder_name)):
        os.mkdir(os.path.join(args.savedir, 'Val',folder_name))
        os.mkdir(os.path.join(args.savedir, 'colored',folder_name))
        os.mkdir(os.path.join(args.savedir, 'overay',folder_name))

    total_time = 0
    for i, imgName in enumerate(image_list):
        img = cv2.imread(imgName)
        if args.overlay:
            img_orig = np.copy(img)

        img = img.astype(np.float32)
        for j in range(3):
            img[:, :, j] -= mean[j]
        for j in range(3):
            img[:, :, j] /= std[j]

        # resize the image to 1024x512x3
        if inputsize != 2048:
            img = cv2.resize(img, (inputsize, inputsize//2))
            if args.overlay:
                img_orig = cv2.resize(img_orig, (inputsize, inputsize//2))

        img /= 255
        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension


        with torch.no_grad():
            img_variable = torch.autograd.Variable(img_tensor)

            if args.gpu:
                img_variable = img_variable.cuda(next(model.parameters()).device)

            start_time = time.time()
            img_out = model(img_variable)
        end_time = time.time() - start_time
        total_time += end_time

        if img_out.size(3) != inputsize:
            factor = inputsize // img_out.size(3)
            img_out = nn.UpsamplingBilinear2d(scale_factor=factor)(img_out)

        classMap_numpy = img_out[0].max(0)[1].cpu().byte().data.numpy()

        if i % 100 == 0:
            print(i)

        name = imgName.split('/')[-1]

        if args.colored:
            classMap_numpy_color = np.zeros((img.shape[1], img.shape[2], img.shape[0]), dtype=np.uint8)
            for idx in range(len(pallete)):
                [r, g, b] = pallete[idx]
                classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
            cv2.imwrite(args.savedir + os.sep + 'colored' + os.sep + folder_name + os.sep + name.replace(args.img_extn, 'png'), classMap_numpy_color)
            if args.overlay:
                overlayed = cv2.addWeighted(img_orig, 0.5, classMap_numpy_color, 0.5, 0)
                cv2.imwrite(args.savedir + os.sep + 'overay' + os.sep + folder_name + os.sep + name.replace(args.img_extn, 'jpg'), overlayed)

        if args.cityFormat:
            classMap_numpy= cv2.resize(classMap_numpy,(2048,1024), interpolation=cv2.INTER_NEAREST)
            classMap_numpy = relabel(classMap_numpy.astype(np.uint8))

        cv2.imwrite(args.savedir + os.sep + 'Val' + os.sep + folder_name + os.sep + name.replace(args.img_extn, 'png'), classMap_numpy)

    print("average FPS {:.2f}".format(len(image_list)/total_time))



def PILevaluateModel(args, model, image_list, folder_name, inputsize):
    # gloabl mean and std values
    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    if not os.path.exists(os.path.join(args.savedir,'Val',folder_name)):
        os.mkdir(os.path.join(args.savedir, 'Val',folder_name))
        os.mkdir(os.path.join(args.savedir, 'colored',folder_name))
        os.mkdir(os.path.join(args.savedir, 'overay',folder_name))

    total_time = 0
    for i, imgName in enumerate(image_list):
        img = Image.open(imgName).convert('RGB')
        if args.overlay:
            img_orig = Image.open(imgName).convert('RGB')
            img_orig = np.array(img_orig)


            # resize the image to 1024x512x3
        if inputsize!=2048:
            img = img.resize((inputsize, inputsize//2), Image.BILINEAR)
        # if args.overlay:
        #     img_orig = cv2.resize(img_orig, (1024, 512))

        img_tensor = F.to_tensor(img)  # convert to tensor (values between 0 and 1)
        img_tensor = F.normalize(img_tensor, mean, std)  # normalize the tensor


        img_tensor = torch.unsqueeze(img_tensor, 0)  # add a batch dimension


        with torch.no_grad():
            img_variable = torch.autograd.Variable(img_tensor)

            if args.gpu:
                img_variable = img_variable.cuda(next(model.parameters()).device)

            start_time = time.time()
            img_out = model(img_variable)
        end_time = time.time() - start_time
        total_time += end_time

        if img_out.size(3) != inputsize:
            factor = inputsize // img_out.size(3)
            img_out = nn.UpsamplingBilinear2d(scale_factor=factor)(img_out)

        classMap_numpy = img_out[0].max(0)[1].cpu().byte().data.numpy()

        if i % 100 == 0:
            print(i)

        name = imgName.split('/')[-1]

        if args.colored:
            classMap_numpy_color = np.zeros((img_orig.shape[0], img_orig.shape[1], img_orig.shape[2]), dtype=np.uint8)
            for idx in range(len(pallete)):
                [r, g, b] = pallete[idx]
                classMap_numpy_color[classMap_numpy == idx] = [b, g, r]
            cv2.imwrite(args.savedir + os.sep + 'colored' + os.sep + folder_name + os.sep + name.replace(args.img_extn, 'png'), classMap_numpy_color)
            if args.overlay:
                overlayed = cv2.addWeighted(img_orig, 0.5, classMap_numpy_color, 0.5, 0)
                cv2.imwrite(args.savedir + os.sep + 'overay' + os.sep + folder_name + os.sep + name.replace(args.img_extn, 'jpg'), overlayed)

        if args.cityFormat:
            if inputsize != 2048:
                classMap_numpy= cv2.resize(classMap_numpy,(2048,1024), interpolation=cv2.INTER_NEAREST)
            classMap_numpy = relabel(classMap_numpy.astype(np.uint8))

        cv2.imwrite(args.savedir + os.sep + 'Val' + os.sep + folder_name + os.sep + name.replace(args.img_extn, 'png'), classMap_numpy)

    print("average FPS {:.2f}".format(len(image_list)/total_time))

def Val_Result(modelname, config, use_nsml, model, mode="CV"):
    ############## save setting
    with open(config) as fin:
        config = json.load(fin)

    data_config = config['data_config']

    savedir = 'Valserver'
    data_dir = data_config['data_dir']
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    savedir = os.path.join(savedir, modelname)

    use_cuda = torch.cuda.is_available()

    if not os.path.isdir(savedir):
        os.mkdir(savedir)
        os.mkdir(os.path.join(savedir, 'colored'))
        os.mkdir(os.path.join(savedir, 'overay'))
        os.mkdir(os.path.join(savedir, 'Val'))
    if use_nsml:
        from nsml import DATASET_PATH
        data_dir = os.path.join(DATASET_PATH, 'train')

    args.config = config
    args.data_dir = os.path.join(data_dir,'leftImg8bit/val')
    args.savedir = savedir
    args.gpu = use_cuda

    # assert args.modelType == 1 and args.decoder, 'Model type should be 2 for ESPNet-C and 1 for ESPNet'
    if args.overlay:
        args.colored = True  # This has to be true if you want to overlay

    #################Make Val image #######################################333
    for i in range(len(Val_folder)):
        print("Val : {}".format(Val_folder[i]))
        full_name = os.path.join(args.data_dir , Val_folder[i])
        image_list = glob.glob(full_name +os.sep + '*.' + args.img_extn)
        print(len(image_list))
        # set to evaluation mode
        model.eval()
        if mode == "CV":
            evaluateModel(args, model, image_list, Val_folder[i], data_config["baseSize"])
        else:
            PILevaluateModel(args, model, image_list, Val_folder[i], data_config["baseSize"])
