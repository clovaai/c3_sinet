import torch
from etc.IOUEval import iouEval
import time
import os
import datetime
import numpy as np

def val(num_gpu, classes, val_loader, model, criterion, outVisdom = False):
    '''
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to evaluation mode
    model.eval()
    rand_pick = np.random.randint(0, len(val_loader))

    iouEvalVal = iouEval(classes)
    total_time =0
    epoch_loss = []
    total_batches = len(val_loader)
    for i, (input, target_coarse, target)in enumerate(val_loader):
        start_time = time.time()

        if num_gpu > 0:
            input = input.cuda()
            target = target.cuda()
            target_coarse = target_coarse.cuda()


        with torch.no_grad():
            input_var, target_var = torch.autograd.Variable(input), torch.autograd.Variable(target)
            target_coarseVar = torch.autograd.Variable(target_coarse)

        # run the mdoel
            output = model(input_var)

            # compute the loss
            loss = criterion(output, target_coarseVar)
            epoch_loss.append(loss.item())

            if outVisdom and i == rand_pick:
                if num_gpu >0:
                    save_input = input_var[0].cpu()
                    save_target = target_var[0].cpu()
                    save_est = output[0].cpu()
                else:
                    save_input = input_var[0]
                    save_target = target_var[0]
                    save_est = output[0]

        time_taken = time.time() - start_time
        total_time += time_taken

        full_out = torch.nn.UpsamplingBilinear2d(size=target_var.size()[1:])(output)

        iouEvalVal.addBatch(full_out.max(1)[1].data, target_var.data)

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()
    print('loss: %.3f time:%.2f' % (average_epoch_loss_val, total_time/total_batches))

    if outVisdom :
        return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU, save_input, save_target, save_est
    else:
        return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU


def train(num_gpu, classes, train_loader, model, criterion, optimizer, epoch, total_ep, aux_w=0.25):
    '''

    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to train mode
    model.train()

    iouEvalTrain = iouEval(classes)

    epoch_loss = []
    epoch_lossC = []

    total_time= 0
    total_batches = len(train_loader)
    for i, (input, target_coarse, target) in enumerate(train_loader):
        start_time = time.time()

        if num_gpu > 0:
            input = input.cuda()
            target = target.cuda()
            target_coarse = target_coarse.cuda()


        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        targetC_var = torch.autograd.Variable(target_coarse)


        # run the mdoel
        output_coarse, output = model(input_var, train=True)

        # set the grad to zero

        loss1 = criterion(output, target_var)
        loss2 = aux_w*criterion(output_coarse, targetC_var)
        loss = loss1+loss2


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss.append(loss.item())
        epoch_lossC.append(loss2.item())


        time_taken = time.time() - start_time
        total_time += time_taken
        # compute the confusion matrix
        iouEvalTrain.addBatch(output.max(1)[1].data, target_var.data)


    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    average_epoch_lossC_train = sum(epoch_lossC) / len(epoch_lossC)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTrain.getMetric()
    print('[%d/%d] loss: %.3f lossC: %.3f time:%.2f'
          % (epoch+1,total_ep, average_epoch_loss_train, average_epoch_lossC_train, total_time/total_batches))

    return average_epoch_loss_train, overall_acc, per_class_acc, per_class_iu, mIOU


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    '''
    helper function to save the checkpoint
    :param state: model state
    :param filenameCheckpoint: where to save the checkpoint
    :return: nothing
    '''
    torch.save(state, filenameCheckpoint)


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


def info_setting(save_dir, model_name):
    now = datetime.datetime.now()
    time_str = now.strftime("%m-%d_%H%M")
    this_savedir = os.path.join(save_dir, model_name+time_str)
    if not os.path.isdir(this_savedir):
        os.mkdir(this_savedir)

    logFileLoc = this_savedir + "/trainValLog.txt"

    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')

        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % (
            'Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val)', 'lr', 'time'))
    return logger, this_savedir
