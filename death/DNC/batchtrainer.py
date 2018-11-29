"""
Trainer E incorporates Channel(). It's a trainer for batch inputs.
"""

import pandas as pd
import torch
import numpy as np
import pdb
from pathlib import Path
import os
from os.path import abspath
from death.post.channelmanager import InputGenD, ChannelManager, train_valid_split
from death.post.inputgen_planE import InputGenE
from torch.utils.data import DataLoader
import torch.nn as nn
from death.DNC.batchDNC import BatchDNC as DNC
from torch.autograd import Variable
import pickle
from shutil import copy
import traceback
from collections import deque
import datetime


batch_size = 1
global_exception_counter = 0
i = None
debug = True
verbose=False

param_x = 69505
param_h = 64
param_L = 4
param_v_t = 5952
param_W = 8
param_R = 8
param_N = 64
param_bs = 128


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
    pickle_file = Path(task_dir).joinpath("saves/" + savestr + "/DNC_" + str(epoch) + "_" + str(iteration) + ".pkl")
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
        "saves/" + savestr + "/DNC_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
    print("loading model at", pickle_file)
    with pickle_file.open('rb') as pickle_file:
        computer, optim, epoch, iteration = torch.load(pickle_file)
    print('Loaded model at epoch ', highestepoch, 'iteartion', iteration)

    return computer, optim, highestepoch, highestiter


def salvage(savestr):
    # this function will pick up the last two highest epoch training and save them somewhere else,
    # this is to prevent unexpected data loss.
    # We are working in a /tmp folder, and we write around 1Gb per minute.
    # The loss of data is likely.

    task_dir = os.path.dirname(abspath(__file__))
    save_dir = Path(task_dir) / "saves" / savestr
    highestepoch = -1
    secondhighestiter = -1
    highestiter = -1
    for child in save_dir.iterdir():
        try:
            epoch = str(child).split("_")[3]
        except IndexError:
            traceback.print_exc()
            print(str(child))
        iteration = str(child).split("_")[4].split('.')[0]
        iteration = int(iteration)
        epoch = int(epoch)
        # some files are open but not written to yet.
        if epoch > highestepoch and iteration > highestiter and child.stat().st_size > 20480:
            highestepoch = epoch
            highestiter = iteration
    if highestepoch == -1 and highestiter == -1:
        print("no file to salvage")
        return
    if secondhighestiter != -1:
        pickle_file2 = Path(task_dir).joinpath(
            "saves/DNC" + savestr + "_" + str(highestepoch) + "_" + str(secondhighestiter) + ".pkl")
        copy(pickle_file2, "/infodev1/rep/projects/jason/pickle/salvage2.pkl")

    pickle_file1 = Path(task_dir).joinpath(
        "saves/DNC" + savestr + "_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
    copy(pickle_file1, "/infodev1/rep/projects/jason/pickle/salvage1.pkl")

    print('salvaged, we can start again with /infodev1/rep/projects/jason/pickle/salvage1.pkl')


def run_one_step(computer, channelmanager, optimizer, binary_criterion):
    computer.train()
    optimizer.zero_grad()
    input, target, loss_type, states_tuple = next(channelmanager)
    target = target.squeeze(1)
    input = Variable(input).cuda()
    target = Variable(target).cuda()
    loss_type = Variable(loss_type).cuda()
    computer.assign_states_tuple(states_tuple)
    output, states_tuple = computer(input)
    channelmanager.push_states(states_tuple)

    time_to_event_output = output[:, 0]
    cause_of_death_output = output[:, 1:]
    time_to_event_target = target[:, 0]
    cause_of_death_target = target[:, 1:]

    loss = binary_criterion(cause_of_death_output, cause_of_death_target)
    loss.backward()
    optimizer.step()
    return loss

def valid_one_step(computer, channelmanager, binary_criterion):
    computer.eval()
    input, target, loss_type, states_tuple = next(channelmanager)
    target = target.squeeze(1)
    input = Variable(input).cuda()
    target = Variable(target).cuda()
    loss_type = Variable(loss_type).cuda()
    computer.assign_states_tuple(states_tuple)
    output, states_tuple = computer(input)
    channelmanager.push_states(states_tuple)

    time_to_event_output = output[:, 0]
    cause_of_death_output = output[:, 1:]
    time_to_event_target = target[:, 0]
    cause_of_death_target = target[:, 1:]

    loss = binary_criterion(cause_of_death_output, cause_of_death_target)
    return loss

def logprint(logfile, string):
    if logprint is not None and logprint !=False:
        with open(logfile, 'a') as handle:
            handle.write(string)
    print(string)

failure=0
def train(computer, optimizer, real_criterion, binary_criterion,
          train, valid, starting_epoch, total_epochs, starting_iter, iter_per_epoch, savestr, logfile=False):
    """

    :param computer:
    :param optimizer:
    :param real_criterion:
    :param binary_criterion:
    :param train: this is the ChannelManager class. It has a __next__ method defined.
    :param valid: ditto
    :param starting_epoch:
    :param total_epochs:
    :param starting_iter:
    :param iter_per_epoch:
    :param savestr: a custom string that identifies this training run
    :param logfile:
    :return:
    """
    global global_exception_counter
    print_interval = 100
    val_interval = 1000
    save_interval = 1000
    target_dim = None
    rldmax_len = 500
    val_batch = 500
    running_loss_deque = deque(maxlen=rldmax_len)

    # erase the logfile
    if logfile:
        open(logfile, 'w').close()
    global i

    for epoch in range(starting_epoch, total_epochs):
        # all these are batches
        for i in range(starting_iter, iter_per_epoch):
            train_step_loss = run_one_step(computer, train, optimizer, binary_criterion)
            if train_step_loss is not None:
                printloss = float(train_step_loss[0])
            else:
                print("What is happening?")
                printloss = 10000
            # computer.new_sequence_reset()
            running_loss_deque.appendleft(printloss)
            if i % print_interval == 0:
                running_loss = np.mean(running_loss_deque)
                if logfile:
                    logprint(logfile, "learning.   count: %4d, training loss: %.10f, running loss: %.10f" %
                                     (i, printloss, running_loss))

            if i % val_interval == 0:
                printloss = 0
                for _ in range(val_batch):
                    val_loss=valid_one_step(computer, valid, binary_criterion)
                    if val_loss is not None:
                        printloss += float(val_loss[0])
                    else:
                        global failure
                        failure+=1
                printloss = printloss / val_batch
                if logfile:
                    logprint(logfile,"validation. count: %4d, val loss     : %.10f" %
                                     (i, printloss))
                else:
                    print("validation. count: %4d, running loss: %.10f" %
                          (i, printloss))

            if i % save_interval == 0:
                save_model(computer, optimizer, epoch, i, savestr)
                print("model saved for epoch", epoch, "input", i)


def valid(computer, optimizer, real_criterion, binary_criterion,
          train, valid, starting_epoch, total_epochs, starting_iter, iter_per_epoch, savestr, logfile=False):
    """
    I have problem comparing the performances of different models. They do not seem to refer to the same value.
    Processing by sequences and processing by steps are fundamentally different and unfair.

    :param computer:
    :param optimizer:
    :param real_criterion:
    :param binary_criterion:
    :param train: this is the ChannelManager class. It has a __next__ method defined.
    :param valid: ditto
    :param starting_epoch:
    :param total_epochs:
    :param starting_iter:
    :param iter_per_epoch:
    :param savestr: a custom string that identifies this training run
    :param logfile:
    :return:
    """
    global global_exception_counter
    print_interval = 100
    val_interval = 10000
    save_interval = 10000
    target_dim = None
    rldmax_len = 500
    val_batch = 100000
    running_loss_deque = deque(maxlen=rldmax_len)
    computer.eval()

    val_losses=[]
    for i in range(val_batch):
        val_loss=valid_one_step(computer, valid, binary_criterion)
        if val_loss is not None:
            printloss = float(val_loss[0])
            val_losses.append(printloss)
        else:
            raise ValueError("Why is val_loss None again?")

        logprint(logfile,"validation. count: %4d, val loss     : %.10f" %
                         (i, printloss))
    print("loss:",np.mean(val_losses))


def forevermain(load=False, lr=1e-3, savestr="", reset=True, palette=False):
    print("Will run main() forever in a loop.")
    while True:
        try:
            main(load, lr, savestr, reset, palette)
        except ValueError:
            traceback.print_exc()

def main(load=True, lr=1e-4, savestr="lowlr"):
    '''
    0.004 is now the new best. But it's not much better. Is lr the problem?
    '''


    '''
    training loss is around 0.0003
    validation loss is around 0.006, peak at 0.005 with early stopping
    still seems overfitting.
    I ran the training for 10 epochs, which took around a week. The running loss keeps decreasing

    epoch   train_loss   valid_loss
    1       0.0018       0.005
    2       0.0016       0.005
    3       0.0012       0.005
    ...
    10      0.0004       0.006

    The log is enclosed for analysis at DNC root folder.
    The validation loss saturates after 10,000 training samples. It suggests inappropriate model or overfitting.

    :param load:
    :param lr:
    :param savestr:
    :return:
    '''
    total_epochs = 10
    iter_per_epoch = 100000
    lr = lr
    optim = None
    starting_epoch = 0
    starting_iteration = 0
    logfile = "log.txt"
    num_workers = 8

    print("Using", num_workers, "workers for training set")
    computer = DNC(x=param_x,
                   h=param_h,
                   L=param_L,
                   v_t=param_v_t,
                   W=param_W,
                   R=param_R,
                   N=param_N,
                   bs=param_bs)

    ig = InputGenD(verbose=False)
    # multiprocessing disabled, because socket request seems unstable.
    # performance should not be too bad?
    trainds, validds = train_valid_split(ig, split_fold=10)
    traindl = DataLoader(dataset=trainds, batch_size=1, num_workers=num_workers)
    validdl = DataLoader(dataset=validds, batch_size=1, num_workers=num_workers)
    traindl = ChannelManager(traindl, param_bs, model=computer)
    validdl = ChannelManager(validdl, param_bs, model=computer)

    # load model:
    if load:
        print("loading model")
        computer, optim, starting_epoch, starting_iteration = load_model(computer, optim, starting_epoch,
                                                                         starting_iteration, savestr)

    computer = computer.cuda()
    if optim is None:
        print("Using Adam with lr", lr)
        optimizer = torch.optim.Adam([i for i in computer.parameters() if i.requires_grad], lr=lr)
    else:
        # print('use Adadelta optimizer with learning rate ', lr)
        # optimizer = torch.optim.Adadelta(computer.parameters(), lr=lr)
        optimizer = optim

    real_criterion = nn.SmoothL1Loss()
    # time-wise sum, label-wise average.
    binary_criterion = nn.BCEWithLogitsLoss()

    # starting with the epoch after the loaded one

    train(computer, optimizer, real_criterion, binary_criterion,
          traindl, validdl, int(starting_epoch), total_epochs, int(starting_iteration), iter_per_epoch, savestr,
          logfile)


def valid_only(savestr="struc"):
    '''

    :return:
    '''

    total_epochs = 10
    iter_per_epoch = 1000000
    lr = None
    optim = None
    starting_epoch = 0
    starting_iteration = 0
    logfile = "log.txt"
    num_workers = 8

    print("Using", num_workers, "workers for training set")
    computer = DNC(x=param_x,
                   h=param_h,
                   L=param_L,
                   v_t=param_v_t,
                   W=param_W,
                   R=param_R,
                   N=param_N,
                   bs=param_bs)

    ig = InputGenD(verbose=verbose, debug=debug)
    # performance should not be too bad?
    trainds, validds = train_valid_split(ig, split_fold=10)
    # change validation set to be inputgenE
    ige=InputGenE()
    validds=ige.get_valid_dataset()

    traindl = DataLoader(dataset=trainds, batch_size=1, num_workers=num_workers)
    validdl = DataLoader(dataset=validds, batch_size=1, num_workers=num_workers)
    traindl = ChannelManager(traindl, param_bs, model=computer)
    validdl = ChannelManager(validdl, param_bs, model=computer)

    # load model:
    print("loading model")
    computer, optim, starting_epoch, starting_iteration = load_model(computer, optim, starting_epoch,
                                                                     starting_iteration, savestr)

    computer = computer.cuda()
    if optim is None:
        print("Using Adam with lr", lr)
        optimizer = torch.optim.Adam([i for i in computer.parameters() if i.requires_grad], lr=lr)
    else:
        # print('use Adadelta optimizer with learning rate ', lr)
        # optimizer = torch.optim.Adadelta(computer.parameters(), lr=lr)
        optimizer = optim

    real_criterion = nn.SmoothL1Loss()
    # time-wise sum, label-wise average.
    binary_criterion = nn.BCEWithLogitsLoss()

    # starting with the epoch after the loaded one

    valid(computer, optimizer, real_criterion, binary_criterion,
          traindl, validdl, int(starting_epoch), total_epochs, int(starting_iteration), iter_per_epoch, savestr,
          logfile)


if __name__ == "__main__":
    valid_only()
