'''
C3SINet
Copyright (c) 2019-present NAVER Corp.
MIT license
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import time
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from etc.Tensor_logger import Logger
from data.dataloader import get_dataloader
import models
from etc.Criteria import CrossEntropyLoss2d
from etc.utils import  *
from etc.VisualizeResults import Vis_Result
from etc.VisualizeValidation import Val_Result

import torchvision

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='./test_setting/SINet.json', help='JSON file for configuration')
    parser.add_argument('-v', '--outvisdom', type=bool, default=False, help='outVisdom')
    parser.add_argument('-t', '--Testserver', type=bool, default=False, help='Testserver')

    args = parser.parse_args()

    ############### setting framework ##########################################
    with open(args.config) as fin:
        config = json.load(fin)
    test_config = config['test_config']
    data_config = config['data_config']

    if data_config["dataset_name"] == "citypilAux":
        from etc.help_functionAux import *
    else:
        from etc.help_function import *

    print("Run : " + test_config["Model"])
    if test_config["Model"].startswith('Dnc_C3'):
        model = models.__dict__[test_config["Model"]](
            classes=test_config["num_classes"], p=test_config["p"], q=test_config["q"],
            D_rate=test_config["D_rate"], chnn=test_config["chnn"])


    elif test_config["Model"].startswith('Dnc_SIN'):
        model = models.__dict__[test_config["Model"]](
            classes=test_config["num_classes"], p=test_config["p"], q=test_config["q"], chnn=test_config["chnn"])

    model_name = test_config["Model"]

    #################### common model setting and opt setting  #######################################



    nsml_logger = Logger(8097, './logs/', False)

    color_transform = Colorize(data_config["classes"])

    start_epoch = 0
    Max_val_iou = 0.0
    Max_name = test_config["model_name"]

    num_gpu = torch.cuda.device_count()

    if num_gpu > 0:
        model.load_state_dict(torch.load(test_config["model_name"]))
    else:
        model.load_state_dict(torch.load(test_config["model_name"], "cpu"))

    if num_gpu:
        print("Use gpu : %d" % num_gpu)
        if num_gpu > 1:
            model = torch.nn.DataParallel(model)
            print("make parallel")
        model = model.cuda()
        print("Done")

    ###################################1stage training models ##############################################

    _, valLoader, data = get_dataloader(data_config)

    weight = torch.from_numpy(data['classWeights'])  # convert the numpy array to torch

    if test_config["num_classes"] == 19:
        weight = weight[:-1]
        criteria = CrossEntropyLoss2d(weight, ignore_id=data_config["ignore_idx"])  # weight

    else:
        weight[-1] = 0
        criteria = CrossEntropyLoss2d(weight)  # weight

    print(weight)
    if num_gpu > 0:
        weight = weight.cuda()
        criteria = criteria.cuda()

    ################################ start Enc train ##########################################
    #
    #
    Val_Result(model_name, args.config, False, model, mode="CV")
    if args.Testserver:
        Vis_Result(model_name, args.config, False, Max_name, mode="CV")

    print("========== validation check ===========")

    # evaluate on validation set
    if args.outvisdom:
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val \
            , save_input, save_target, save_est = \
            val(num_gpu, data_config["classes"], valLoader, model, criteria, args.outvisdom)

        grid_targets = torchvision.utils.make_grid(color_transform(save_est.max(1)[1].data), nrow=6)
        nsml_logger.image_summary(grid_targets,
                                  opts=dict(title=f'VAL est (epoch: {0}, IOU: {str(mIOU_val)})',
                                            caption=f'VAL est(epoch: {0},IOU: {str(mIOU_val)})', ))

        grid_gt = torchvision.utils.make_grid(color_transform(save_target.data), nrow=6)
        nsml_logger.image_summary(grid_gt,
                                  opts=dict(title=f'VAL gt (epoch: {0}, step: {str(mIOU_val)})',
                                            caption=f'VAL gt (epoch: {0}, step: {str(mIOU_val)})', ))
    else:
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = \
            val(num_gpu, data_config["classes"], valLoader, model, criteria)
    print("Val Loss = %.4f\t mIOU = %.4f\t Acc = %.4f \n" % (lossVal, mIOU_val, overall_acc_val))









