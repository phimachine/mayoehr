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
from death.post.inputgen_planF import InputGenF
from torch.utils.data import DataLoader
from death.post.channelmanager import ChannelManager
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

param_x = 66529
param_h = 4 #64
param_L = 2
param_v_t = 5952
param_W = 2 #8
param_R = 2 #8
param_N = 4 #64
param_bs = 128

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def save_model(net, optim, epoch, iteration, savestr):
    epoch = int(epoch)
    task_dir = os.path.dirname(abspath(__file__))
    if not os.path.exists(Path(task_dir) / "saves" / savestr):
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
    print('Loaded model at epoch ', highestepoch, 'iteration', iteration)

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
    string=str(string)
    if logprint is not None and logprint !=False:
        with open(logfile, 'a') as handle:
            handle.write(string+'\n')
    print(string)

failure=0
def train(computer, optimizer, real_criterion, binary_criterion,
          ig, validdl, starting_epoch, total_epochs, starting_iter, iter_per_epoch, savestr,
          num_workers, logfile=False):
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
        open(logfile, 'w+').close()
    global i

    for epoch in range(starting_epoch, total_epochs):
        train=ig.get_train()
        traindl = DataLoader(dataset=train, batch_size=1, num_workers=num_workers)
        traincm = ChannelManager(traindl, param_bs, model=computer)

        # all these are batches
        for i in range(starting_iter, iter_per_epoch):
            train_step_loss = run_one_step(computer, traincm, optimizer, binary_criterion)
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
                    val_loss=valid_one_step(computer, validdl, binary_criterion)
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
        starting_iter=0


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

    ig = InputGenF(death_fold=0)
    trainds = ig.get_train_dataset()
    validds = ig.get_valid_dataset()
    testds = ig.get_test_dataset()
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

def main(savestr, load=False, lr=1e-4,curri=False):
    '''
    11/28
    0.004 is now the new best. But it's not much better. Is lr the problem?

    11/29
    lr does not seem to be the problem
    Changed background distribution. Loss is now around 0.001. Training loss is 10 times smaller.
    '''

    '''
    12/2
    Can I vary the parameters to an extreme so that the running loss is close to the validation?
    If somehow this is impossible, the only reason I can think of is backpropagation leaking.
    Because Channelmanager evaluates at every timestamp, backpropagation alters the weights at each step.
    Somehow, that can be signals of the final prediction, which is utilized in the next step's prediction.
    Backpropagation on each timesteps leaks the final label.
    This is interesting. 
    
    Running loss typically at 0.0002. Validation loss at 0.0012.
    '''

    '''
    Running 0.0003. Validation 0.0007.
    Parameters:
    param_x = 69505
    param_h = 2 #64
    param_L = 2
    param_v_t = 5952
    param_W = 2 #8
    param_R = 2 #8
    param_N = 2 #64
    param_bs = 128
    '''

    '''
    slight increase in validation with increased h and N.
    Running 0.0003, Validation 0.0008
    '''
    np.warnings.filterwarnings('ignore')

    total_epochs = 3
    iter_per_epoch = 10000
    lr = lr
    print("Using lr=",lr)
    optim = None
    starting_epoch = 0
    starting_iteration = 0
    logstring = str(datetime.datetime.now().time())
    logstring.replace(" ", "_")
    logfile = "log/"+savestr+"_"+logstring+".txt"
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
    if curri:
        death_fold=5
    else:
        death_fold=0
    ig = InputGenF(death_fold=death_fold,curriculum=curri)


    validds = ig.get_valid()
    testds = ig.get_test()
    validdl = DataLoader(dataset=validds, batch_size=1, num_workers=num_workers)
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
          ig, validdl, int(starting_epoch), total_epochs, int(starting_iteration), iter_per_epoch, savestr,
          num_workers, logfile)




if __name__ == "__main__":
    main(savestr="small",curri=False)
