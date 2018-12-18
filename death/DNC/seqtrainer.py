import pandas as pd
import torch
import numpy as np
import pdb
from pathlib import Path
import os
from os.path import abspath
from death.post.inputgen_planG import InputGenG, pad_collate
from death.post.inputgen_planH import InputGenH
from death.DNC.seqDNC import SeqDNC
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.modules import LSTM
from torch.autograd import Variable
import pickle
from shutil import copy
import traceback
from collections import deque
import datetime
from death.DNC.batchtrainer import logprint
import pdb

param_x = 66529
param_h = 64  # 64
param_L = 4  # 4
param_v_t = 5952
param_W = 8  # 8
param_R = 8  # 8
param_N = 64  # 64
param_bs = 8

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def datetime_filename():
    return datetime.datetime.now().strftime("%m_%d_%X")


def sv(var):
    return var.data.cpu().numpy()


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def save_model(net, optim, epoch, iteration, savestr):
    epoch = int(epoch)
    task_dir = os.path.dirname(abspath(__file__))
    if not os.path.isdir(Path(task_dir) / "saves" / savestr):
        os.mkdir(Path(task_dir) / "saves" / savestr)
    pickle_file = Path(task_dir).joinpath("saves/" + savestr + "/seqDNC_" + str(epoch) + "_" + str(iteration) + ".pkl")
    with pickle_file.open('wb') as fhand:
        torch.save((net, optim, epoch, iteration), fhand)
    print('model saved at', pickle_file)


def load_model(computer, optim, starting_epoch, starting_iteration, savestr):
    task_dir = os.path.dirname(abspath(__file__))
    save_dir = Path(task_dir) / "saves" / savestr
    highestepoch = 0
    highestiter = 0
    for child in save_dir.iterdir():
        try:
            epoch = str(child).split("_")[3]
            iteration = str(child).split("_")[4].split('.')[0]
        except IndexError:
            print(str(child))
        iteration = int(iteration)
        epoch = int(epoch)
        # some files are open but not written to yet.
        if child.stat().st_size > 20480:
            if epoch > highestepoch or (iteration > highestiter and epoch == highestepoch):
                highestepoch = epoch
                highestiter = iteration
    if highestepoch == 0 and highestiter == 0:
        print("nothing to load")
        return computer, optim, starting_epoch, starting_iteration
    pickle_file = Path(task_dir).joinpath(
        "saves/" + savestr + "/seqDNC_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
    print("loading model at", pickle_file)
    with pickle_file.open('rb') as pickle_file:
        computer, optim, epoch, iteration = torch.load(pickle_file)
    print('Loaded model at epoch ', highestepoch, 'iteration', iteration)

    return computer, optim, highestepoch, highestiter

global_exception_counter = 0


def run_one_patient(computer, input, target, target_dim, optimizer, loss_type, real_criterion,
                    binary_criterion, validate=False):
    global global_exception_counter
    patient_loss = None
    try:
        if not validate:
            computer.train()
        else:
            computer.eval()
        optimizer.zero_grad()
        input = Variable(torch.Tensor(input).cuda())
        target = Variable(torch.Tensor(target).cuda())

        # we have no critical index, becuase critical index are those timesteps that
        # DNC is required to produce outputs. This is not the case for our project.
        # criterion does not need to be reinitiated for every story, because we are not using a mask

        patient_output = computer(input)
        cause_of_death_output = patient_output[:, 1:]
        cause_of_death_target = target[:, 1:]
        # pdb.set_trace()
        patient_loss = binary_criterion(cause_of_death_output, cause_of_death_target)

        if not validate:
            patient_loss.backward()
            optimizer.step()

        if global_exception_counter > -1:
            global_exception_counter -= 1
    except ValueError:
        traceback.print_exc()
        print("Value Error reached")
        print(datetime.datetime.now().time())
        global_exception_counter += 1
        if global_exception_counter == 10:
            save_model(computer, optimizer, epoch=0, iteration=global_exception_counter)
            raise ValueError("Global exception counter reached 10. Likely the model has nan in weights")
        else:
            pass

    return patient_loss


def train(computer, optimizer, real_criterion, binary_criterion,
          train, valid_dl, starting_epoch, total_epochs, starting_iter, iter_per_epoch, savestr, logfile=False):
    valid_iterator = iter(valid_dl)
    print_interval = 10
    val_interval = 400
    save_interval = 800
    target_dim = None
    rldmax_len = 50
    val_batch = 100
    running_loss_deque = deque(maxlen=rldmax_len)
    if logfile:
        open(logfile, 'w').close()

    for name, param in computer.named_parameters():
        logprint(logfile, name)
        logprint(logfile, param.data.shape)

    for epoch in range(starting_epoch, total_epochs):
        for i, (input, target, loss_type) in enumerate(train):
            i = starting_iter + i
            if target_dim is None:
                target_dim = target.shape[1]

            if i < iter_per_epoch:
                train_story_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                                   real_criterion, binary_criterion)
                if train_story_loss is not None:
                    printloss = float(train_story_loss[0])
                else:
                    raise ValueError("Why would story loss be None?")
                running_loss_deque.appendleft(printloss)
                if i % print_interval == 0:
                    running_loss = np.mean(running_loss_deque)
                    logprint(logfile, "learning.   count: %4d, training loss: %.10f, running loss: %.10f" %
                             (i, printloss, running_loss))

                if i % val_interval == 0:
                    printloss = 0
                    for _ in range(val_batch):
                        # we should consider running validation multiple times and average. TODO
                        try:
                            (input, target, loss_type) = next(valid_iterator)
                        except StopIteration:
                            valid_iterator = iter(valid_dl)
                            (input, target, loss_type) = next(valid_iterator)

                        val_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                                   real_criterion, binary_criterion, validate=True)
                        if val_loss is not None:
                            printloss += float(val_loss[0])
                        else:
                            raise ValueError("Investigate this")
                    printloss = printloss / val_batch
                    # TODO this validation is not printing correctly. Way too big.
                    logprint(logfile, "validation. count: %4d, val loss     : %.10f" %
                             (i, printloss))

                if i % save_interval == 0:
                    save_model(computer, optimizer, epoch, i, savestr)
                    print("model saved for epoch", epoch, "input", i)
            else:
                break


def valid(computer, optimizer, real_criterion, binary_criterion,
          train, valid_dl, starting_epoch, total_epochs, starting_iter, iter_per_epoch, logfile=False):
    running_loss = []
    target_dim = None
    valid_iterator = iter(valid_dl)

    for i in valid_iterator:
        input, target, loss_type = next(valid_iterator)
        val_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                   real_criterion, binary_criterion, validate=True)
        if val_loss is not None:
            printloss = float(val_loss[0])
            running_loss.append((printloss))
        if logfile:
            with open(logfile, 'a') as handle:
                handle.write("validation. count: %4d, val loss     : %.10f \n" %
                             (i, printloss))
        print("validation. count: %4d, val loss: %.10f" %
              (i, printloss))
    print(np.mean(running_loss))



def validationonly(savestr):
    '''
    :return:
    '''

    lr = 1e-3
    optim = None
    logfile = "vallog.txt"

    num_workers = 8
    ig = InputGenH()
    # multiprocessing disabled, because socket request seems unstable.
    # performance should not be too bad?
    validds=ig.get_valid()
    validdl = DataLoader(dataset=validds,num_workers=num_workers, batch_size=param_bs, collate_fn=pad_collate)
    print("Using", num_workers, "workers for validation set")
    computer = SeqDNC(x=param_x,
                      h=param_h,
                      L=param_L,
                      v_t=param_v_t,
                      W=param_W,
                      R=param_R,
                      N=param_N,
                      bs=param_bs)
    # load model:
    print("loading model")
    computer, optim, starting_epoch, starting_iteration = load_model(computer, optim, 0, 0, savestr)

    computer = computer.cuda()
    optimizer=None

    real_criterion = nn.SmoothL1Loss()
    binary_criterion = nn.BCEWithLogitsLoss()

    traindl=None
    total_epochs=None
    iter_per_epoch=None

    # starting with the epoch after the loaded one
    valid(computer, optimizer, real_criterion, binary_criterion,
          traindl, validdl, int(starting_epoch), total_epochs,int(starting_iteration), iter_per_epoch, logfile)

def main(load, savestr='default', lr=1e-3, curri=False):
    """
    :param load:
    :param savestr:
    :param lr:
    :param curri:
    :return:
    """


    total_epochs = 5
    iter_per_epoch = 100000
    optim = None
    starting_epoch = 0
    starting_iteration = 0
    logfile = "log/" + savestr + "_" + datetime_filename() + ".txt"

    num_workers = 16
    ig = InputGenG()
    trainds = ig.get_train()
    validds = ig.get_valid()
    testds = ig.get_test()
    traindl = DataLoader(dataset=trainds, batch_size=param_bs, num_workers=num_workers, collate_fn=pad_collate)
    validdl = DataLoader(dataset=validds, batch_size=param_bs, num_workers=num_workers//2, collate_fn=pad_collate)

    print("Using", num_workers, "workers for training set")
    computer = SeqDNC(x=param_x,
                      h=param_h,
                      L=param_L,
                      v_t=param_v_t,
                      W=param_W,
                      R=param_R,
                      N=param_N,
                      bs=param_bs)
    # load model:
    if load:
        print("loading model")
        computer, optim, starting_epoch, starting_iteration = load_model(computer, optim, starting_epoch,
                                                                         starting_iteration, savestr)

    computer = computer.cuda()
    if optim is None:
        optimizer = torch.optim.Adam(computer.parameters(), lr=lr)
    else:
        # print('use Adadelta optimizer with learning rate ', lr)
        # optimizer = torch.optim.Adadelta(computer.parameters(), lr=lr)
        optimizer = optim
        for group in optimizer.param_groups:
            print("Currently using a learing rate of ", group["lr"])


    real_criterion = nn.SmoothL1Loss()
    binary_criterion = nn.BCEWithLogitsLoss()

    # starting with the epoch after the loaded one

    train(computer, optimizer, real_criterion, binary_criterion,
          traindl, validdl, int(starting_epoch), total_epochs,
          int(starting_iteration), iter_per_epoch, savestr, logfile)


if __name__ == "__main__":
    # main(load=True
    #main(False)
    validationonly("retrain")


    """
    12/6
    0.0003 validation at around 8000 batches, equal to the running loss.
    Note that after 400 batches, it's already around 0.0004. This is important, because that allows me to use early stopping.
    Interesting. Now I vary the hyperparameter for more tests. Try radically decrease.
    Increase of parameter size might work in favor, because overfitting is not a severe issue.
    Small variance means I can probably lower the learning rate and feed with more data.
    """

    """
    12/7
    0.0004 is the average validation for a smaller parameter size. No batch validation reached below 0.0003.
    For old parameter set, it happened 3 or 4 times. Pretty significant.
    It was reached pretty early too.
    Variance of both results cannot be identified with my eyes. 
    The next step is to increase parameter set and lower lr, train for a long time.
    """

    """
    12/17
    Loading /infodev1 bottlenecks. Not sure.
    Lost the log. Somehow validation has problems. Training becomes very slow. What's happening? 
    I did not change anything. Sync?
    """