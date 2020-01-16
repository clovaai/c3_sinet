'''
C3SINet
Copyright (c) 2019-present NAVER Corp.
MIT license
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
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
from etc.help_function import *
from etc.utils import  *
from etc.VisualizeResults import Vis_Result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='./setting/C3NetV2.json', help='JSON file for configuration')
    parser.add_argument('-d', '--decoder_only', type=bool, default=False, help='Decoder only training')
    # parser.add_argument('-o', '--optim', type=str, default="Adam", help='Adam , SGD, RMS')
    # parser.add_argument('-s', '--lrsch', type=str, default="multistep", help='step, poly, multistep, warmpoly')
    # parser.add_argument('-t', '--wd_tfmode', type=bool, default=True, help='Play with NSML!')
    # parser.add_argument('-w', '--weight_decay', type=float, default=2e-4, help='value for weight decay')


    args = parser.parse_args()

    ############### setting framework ##########################################


    with open(args.config) as fin:
        config = json.load(fin)
    train_config = config['train_config']
    data_config = config['data_config']
    # [2,3,7,13] [2,4,8,16]

    args.optim = train_config["optim"]
    args.lrsch = train_config["lrsch"]
    args.wd_tfmode = train_config["wd_tfmode"]
    args.weight_decay = train_config["weight_decay"]
    others = args.weight_decay * train_config["tf_decay"]

    if not os.path.isdir(train_config['save_dir']):
        os.mkdir(train_config['save_dir'])

    print("Run : " + train_config["Model"])
    if train_config["Model"].startswith('Enc_C3'):
        model = models.__dict__[train_config["Model"]]( classes=train_config["num_classes"],
             p=train_config["p"], q=train_config["q"], D_rate = train_config["D_rate"], chnn = train_config["chnn"])



    model_name = train_config["Model"]


    #################### common model setting and opt setting  #######################################

    nsml_logger = Logger(8097, './logs', False)


    start_epoch = 0
    Max_val_iou = 0.0
    Max_name = ''

    if train_config["resume"]:
        if os.path.isfile(train_config["resume"]):
            print("=> loading checkpoint '{}'".format(train_config["resume"]))
            checkpoint = torch.load(train_config["resume"])
            start_epoch = checkpoint['epoch']
            # args.lr = checkpoint['lr']
            Max_name = checkpoint['Max_name']
            Max_val_iou = checkpoint['Max_val_iou']

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(train_config["resume"], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(train_config["resume"]))

    num_gpu = torch.cuda.device_count()
    if num_gpu > 0:
        print("Use gpu : %d" % num_gpu)
        if num_gpu > 1:
            model = torch.nn.DataParallel(model)
            print("make parallel")
        model = model.cuda()
        print("GPU setting Done")
  ###################################stage Enc setting ##############################################
    if (not args.decoder_only):
        logger, this_savedir = info_setting(train_config['save_dir'], train_config["Model"])
        logger.flush()
        logdir = this_savedir.split(train_config['save_dir'])[1]

        trainLoader, trainLoader_scale1, trainLoader_scale2, trainLoader_scale3, trainLoader_scale4, valLoader, data \
            = get_dataloader(data_config)

        weight = torch.from_numpy(data['classWeights'])  # convert the numpy array to torch
        weight[-1] = 0
        if num_gpu > 0:
            weight = weight.cuda()
        print(weight)
        criteria = CrossEntropyLoss2d(weight)
        if num_gpu > 0:
            criteria = criteria.cuda()

        params_set = []
        names_set = []

        if args.wd_tfmode:
            params_dict = dict(model.named_parameters())
            for key, value in params_dict.items():
                if len(value.data.shape) == 4:
                    if value.data.shape[1] == 1:
                        params_set += [{'params': [value], 'weight_decay': 0.0}]
                        # names_set.append(key)
                    else:
                        params_set += [{'params': [value], 'weight_decay': args.weight_decay}]
                else:
                    params_set += [{'params': [value], 'weight_decay': others}]
            #
            #         if "bn" in key:
            #             if "weight" in key:
            #                 params_set += [{'params': [value], 'weight_decay': others}]
            #                 names_set.append(key)
            #             else:
            #                 params_set += [{'params': [value], 'weight_decay': 0.0}]
            #         else:
            #             params_set += [{'params': [value], 'weight_decay': 0.0}]
            # print(names_set)

            if args.optim == "Adam":
                optimizer = torch.optim.Adam(params_set, train_config['learning_rate'], (0.9, 0.999), eps=1e-08,
                                             weight_decay=args.weight_decay)
            elif args.optim == "SGD":
                optimizer = torch.optim.SGD(params_set, train_config["learning_rate"], momentum=0.9,
                                            weight_decay=args.weight_decay, nesterov=True)
            elif args.optim == "RMS":
                optimizer = torch.optim.RMSprop(params_set, train_config["learning_rate"], alpha=0.9, momentum=0.9,
                                                eps=1, weight_decay=args.weight_decay)

        else:
            if args.optim == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), train_config['learning_rate'], (0.9, 0.999), eps=1e-08,
                                             weight_decay=args.weight_decay)
            elif args.optim == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), train_config["learning_rate"], momentum=0.9,
                                            weight_decay=args.weight_decay, nesterov=True)
            elif args.optim == "RMS":
                optimizer = torch.optim.RMSprop(model.parameters(), train_config["learning_rate"], alpha=0.9,
                                                momentum=0.9, eps=1, weight_decay=args.weight_decay)
        # print(str(optimizer))

        init_lr = train_config["learning_rate"]
        if args.lrsch == "multistep":
            decay1 = train_config["epochs"] // 2
            decay2 = train_config["epochs"] - train_config["epochs"] // 6
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[decay1, decay2], gamma=0.5)
        elif args.lrsch == "step":
            step = train_config["epochs"] // 3
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)
        elif args.lrsch == "poly":
            lambda1 = lambda epoch: pow((1 - ((epoch - 1) / train_config["epochs"])), 0.98)  ## scheduler 2
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  ## scheduler 2
        elif args.lrsch == "warmpoly":
            scheduler = WarmupPoly(init_lr=init_lr, total_ep=train_config["epochs"])


        print("init_lr: " + str(train_config["learning_rate"]) + "   batch_size : " + str(data_config["batch_size"]) +
              "\t" + args.lrsch + " sch use weight and class " + str(train_config["num_classes"]))
        print("logs saved in " + logdir + "\tlr sch: " + args.lrsch + "\toptim method: " + args.optim +
              "\ttf style : " + str(args.wd_tfmode) + "\tbn-weight : " + str(others))


    ################################ start Enc train ##########################################


        print("========== ENCODER TRAINING ===========")
        for epoch in range(start_epoch, train_config["epochs"]):
            if args.lrsch == "poly":
                scheduler.step(epoch)  ## scheduler 2
            else:
                scheduler.step()

            lr = 0
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print("Learning rate: " + str(lr))

            # train for one epoch
            # We consider 1 epoch with all the training data (at different scales)
            start_t = time.time()
            train(num_gpu, data_config["classes"], trainLoader_scale1, model, criteria, optimizer, epoch, train_config["epochs"])
            train(num_gpu, data_config["classes"], trainLoader_scale2, model, criteria, optimizer, epoch, train_config["epochs"])
            train(num_gpu, data_config["classes"], trainLoader_scale4, model, criteria, optimizer, epoch, train_config["epochs"])
            train(num_gpu, data_config["classes"], trainLoader_scale3, model, criteria, optimizer, epoch, train_config["epochs"])

            lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr = \
                train(num_gpu, data_config["classes"], trainLoader, model, criteria, optimizer, epoch, train_config["epochs"])

            # evaluate on validation set
            lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = \
                val(num_gpu, data_config["classes"], valLoader, model, criteria)

            end_t = time.time()

            if num_gpu > 1:
                this_state_dict = model.module.state_dict()
            else:
                this_state_dict = model.state_dict()

            if epoch >= train_config["epochs"]*0.6 :
                model_file_name = this_savedir + '/model_' + str(epoch + 1) + '.pth'
                torch.save(this_state_dict, model_file_name)

                if (Max_val_iou < mIOU_val):
                    Max_val_iou = mIOU_val
                    Max_name = model_file_name
                    print(" new max iou : " + Max_name + '\t' + str(mIOU_val))
                    model_best_name = this_savedir + '/bestmodel' + '.pth'
                    torch.save(this_state_dict, model_best_name)

                with open(this_savedir + '/acc_' + str(epoch + 1) + '.txt', 'w') as log:
                    log.write(
                        "\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (
                            epoch+1, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
                    log.write('\n')
                    log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
                    log.write('\n')
                    log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
                    log.write('\n')
                    log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
                    log.write('\n')
                    log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))


            logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f\t\t%.2f" % (
                epoch+1, lossTr, lossVal, mIOU_tr, mIOU_val, lr, (end_t - start_t)))
            logger.flush()

            save_checkpoint({
                'epoch': epoch + 1, 'arch': str(model),
                'state_dict': this_state_dict,
                'optimizer': optimizer.state_dict(),
                'lossTr': lossTr, 'lossVal': lossVal,
                'iouTr': mIOU_tr, 'iouVal': mIOU_val,
                'lr': lr,
                'Max_name': Max_name, 'Max_val_iou': Max_val_iou
            }, this_savedir + '/checkpoint.pth.tar')

            print("Epoch : " + str(epoch+1) + ' Details')
            print("Epoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f \n" % (
                epoch+1, lossTr, lossVal, mIOU_tr, mIOU_val))

            info = {
                'S1_train_loss': lossTr,
                'S1_val_loss': lossVal,
                'S1_train_iou': mIOU_tr,
                'S1_val_iou': mIOU_val,
                'S1_lr': lr
            }
            for tag, value in info.items():
                nsml_logger.scalar_summary(tag, value, epoch + 1)
        logger.close()

        print(" Enc max iou : " + Max_name + '\t' + str(Max_val_iou))

        #########################################---Decoder---##################################################

        print("get max iou file : " + Max_name)
        if model_name.startswith('Enc'):
            model_name = "Dnc" + train_config["Model"].split('Enc')[1]

        if model_name.startswith('Dnc_C3'):
            model = models.__dict__[model_name](
                classes=train_config["num_classes"], p=train_config["p"], q=train_config["q"],
                D_rate=train_config["D_rate"], chnn=train_config["chnn"], encoderFile=Max_name)


        else:
            print(model_name + " \t wrong model name")
            exit(0)

        if num_gpu > 0:
            if num_gpu >1:
                model = torch.nn.DataParallel(model)
                print("Make Data parallel")
            model = model.cuda()
            cudnn.benchmark = True

    start_epoch = 0
    Max_val_iou = 0.0
    Max_name = ''
    logger, this_savedir = info_setting(train_config['save_dir'], model_name)
    logger.flush()
    logdir = this_savedir.split(train_config['save_dir'])[1]

    data_config["batch_size"] = train_config["batch_size2"]
    data_config["scaleIn"] = train_config["scaleIn2"]
    data_config["num_work"] = 4

    trainLoader, trainLoader_scale1, trainLoader_scale2, trainLoader_scale3, trainLoader_scale4, valLoader, data \
        = get_dataloader(data_config)

    weight = torch.from_numpy(data['classWeights'])  # convert the numpy array to torch
    weight[-1] = 0
    if num_gpu > 0:
        weight = weight.cuda()
    print(weight)
    criteria = CrossEntropyLoss2d(weight)

    if num_gpu > 0:
        criteria = criteria.cuda()

    params_set = []
    names_set = []

    # others = others*2
    if args.wd_tfmode:
        params_dict = dict(model.named_parameters())
        for key, value in params_dict.items():
            if len(value.data.shape) == 4:
                if value.data.shape[1] == 1:
                    params_set += [{'params': [value], 'weight_decay': 0.0}]
                    # names_set.append(key)
                else:
                    params_set += [{'params': [value], 'weight_decay': args.weight_decay}]
            else:
                params_set += [{'params': [value], 'weight_decay': others}]

        #         if "bn" in key:
        #             if "weight" in key:
        #                 params_set += [{'params': [value], 'weight_decay': others}]
        #                 names_set.append(key)
        #             else:
        #                 params_set += [{'params': [value], 'weight_decay': 0.0}]
        #         else:
        #             params_set += [{'params': [value], 'weight_decay': 0.0}]
        # print(names_set)

        if args.optim == "Adam":
            optimizer = torch.optim.Adam(params_set, train_config['learning_rate2'], (0.9, 0.999), eps=1e-08,
                                         weight_decay=args.weight_decay)
        elif args.optim == "SGD":
            optimizer = torch.optim.SGD(params_set, train_config["learning_rate2"], momentum=0.9,
                                        weight_decay=args.weight_decay, nesterov=True)
        elif args.optim == "RMS":
            optimizer = torch.optim.RMSprop(params_set, train_config["learning_rate2"], alpha=0.9, momentum=0.9,
                                            eps=1, weight_decay=args.weight_decay)

    else:
        if args.optim == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), train_config['learning_rate2'], (0.9, 0.999), eps=1e-08,
                                         weight_decay=args.weight_decay)
        elif args.optim == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), train_config["learning_rate2"], momentum=0.9,
                                        weight_decay=args.weight_decay, nesterov=True)
        elif args.optim == "RMS":
            optimizer = torch.optim.RMSprop(model.parameters(), train_config["learning_rate2"], alpha=0.9,
                                            momentum=0.9, eps=1, weight_decay=args.weight_decay)
    # print(str(optimizer))

    init_lr = train_config["learning_rate2"]
    if args.lrsch == "multistep":
        decay1 = train_config["epochs"] // 2
        decay2 = train_config["epochs"] - train_config["epochs"] // 6
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[decay1, decay2], gamma=0.5)
    elif args.lrsch == "step":
        step = train_config["epochs"] // 3
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)
    elif args.lrsch == "poly":
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / train_config["epochs"])), 0.98)  ## scheduler 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  ## scheduler 2
    elif args.lrsch == "warmpoly":
        scheduler = WarmupPoly(init_lr=init_lr, total_ep=train_config["epochs"])

    print("init_lr: " + str(train_config["learning_rate2"]) + "   batch_size : " + str(data_config["batch_size"]) +
          "\t" + args.lrsch + " sch use weight and class " + str(train_config["num_classes"]))
    print("logs saved in " + logdir + "\tlr sch: " + args.lrsch + "\toptim method: " + args.optim +
          "\ttf style : " + str(args.wd_tfmode) + "\tbn-weight : " + str(others))

###################################---start Dnc train-----###################################################

    print("========== DECODER TRAINING ===========")

        # When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    for epoch in range(start_epoch, train_config["epochs"]):

        # scheduler.step(epoch)  ## scheduler 2
        if args.lrsch == "poly":
            scheduler.step(epoch)  ## scheduler 2
        else:
            scheduler.step()
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))

        # train for one epoch
        # We consider 1 epoch with all the training data (at different scales)
        start_t = time.time()
        train(num_gpu, data_config["classes"], trainLoader_scale1, model, criteria, optimizer, epoch, train_config["epochs"])
        train(num_gpu, data_config["classes"], trainLoader_scale2, model, criteria, optimizer, epoch, train_config["epochs"])
        train(num_gpu, data_config["classes"], trainLoader_scale4, model, criteria, optimizer, epoch, train_config["epochs"])
        train(num_gpu, data_config["classes"], trainLoader_scale3, model, criteria, optimizer, epoch, train_config["epochs"])
        lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr = \
            train(num_gpu, data_config["classes"], trainLoader, model, criteria, optimizer, epoch, train_config["epochs"])

        # evaluate on validation set

        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = \
            val(num_gpu, data_config["classes"], valLoader, model, criteria)

        end_t = time.time()
        if num_gpu>1:
            this_state_dict = model.module.state_dict()
        else:
            this_state_dict = model.state_dict()

        # save the model also


        if epoch >= train_config["epochs"]*0.65 :
            model_file_name = this_savedir + '/model_' + str(epoch + 1) + '.pth'
            torch.save(this_state_dict, model_file_name)

            if (Max_val_iou < mIOU_val):
                Max_val_iou = mIOU_val
                Max_name = model_file_name
                print(" new max iou : " + Max_name + '\t' + str(mIOU_val))
                model_best_name = this_savedir + '/bestmodel' + '.pth'
                torch.save(this_state_dict, model_best_name)

            with open(this_savedir + '/acc_' + str(epoch + 1) + '.txt', 'w') as log:
                log.write(
                    "\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (
                        epoch+1, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
                log.write('\n')
                log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
                log.write('\n')
                log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
                log.write('\n')
                log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
                log.write('\n')
                log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f\t\t%.2f" % (
            epoch+1, lossTr, lossVal, mIOU_tr, mIOU_val, lr, (end_t - start_t)))
        logger.flush()

        save_checkpoint({
            'epoch': epoch + 1, 'arch': str(model),
            'state_dict': this_state_dict,
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr, 'lossVal': lossVal,
            'iouTr': mIOU_tr, 'iouVal': mIOU_val,
            'lr': lr,
            'Max_name': Max_name, 'Max_val_iou': Max_val_iou
        }, this_savedir + '/checkpoint.pth.tar')

        print("Epoch : " + str(epoch+1) + ' Details')
        print("Epoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f \n" % (
            epoch+1, lossTr, lossVal, mIOU_tr, mIOU_val))

        info = {
            'train_loss': lossTr,
            'val_loss': lossVal,
            'train_iou': mIOU_tr,
            'val_iou': mIOU_val,
            'lr': lr
        }
        for tag, value in info.items():
            nsml_logger.scalar_summary(tag, value, epoch + 1)

    logger.close()
    print(" new max iou : " + Max_name + '\t' + str(Max_val_iou))

    print("========== TRAINING FINISHED ===========")

    Vis_Result(model_name, args.config, False, Max_name)







