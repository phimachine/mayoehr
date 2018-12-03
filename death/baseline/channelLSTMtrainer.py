import pandas as pd
import torch
import numpy as np
import pdb
from pathlib import Path
import os
from os.path import abspath
from death.post.inputgen_planF import InputGenF
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.modules import LSTM
from torch.autograd import Variable
import pickle
from shutil import copy
import traceback
from collections import deque
import datetime
from death.baseline.channelLSTM import ChannelLSTM
from death.baseline.lstmcm import ChannelManager

batch_size = 1


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
    pickle_file = Path(task_dir).joinpath("saves/" + savestr + "/lstm_" + str(epoch) + "_" + str(iteration) + ".pkl")
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
        "saves/" + savestr + "/lstm_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
    print("loading model at", pickle_file)
    with pickle_file.open('rb') as pickle_file:
        computer, optim, epoch, iteration = torch.load(pickle_file)
    print('Loaded model at epoch ', highestepoch, 'iteartion', iteration)

    return computer, optim, highestepoch, highestiter


def salvage():
    # this function will pick up the last two highest epoch training and save them somewhere else,
    # this is to prevent unexpected data loss.
    # We are working in a /tmp folder, and we write around 1Gb per minute.
    # The loss of data is likely.

    task_dir = os.path.dirname(abspath(__file__))
    save_dir = Path(task_dir) / "lstmsaves"
    highestepoch = -1
    secondhighestiter = -1
    highestiter = -1
    for child in save_dir.iterdir():
        epoch = str(child).split("_")[3]
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
        pickle_file2 = Path(task_dir).joinpath("lstmsaves/lstm_" + str(highestepoch) + "_" + str(secondhighestiter) + ".pkl")
        copy(pickle_file2, "/infodev1/rep/projects/jason/pickle/lstmsalvage2.pkl")

    pickle_file1 = Path(task_dir).joinpath("lstmsaves/lstm_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
    copy(pickle_file1, "/infodev1/rep/projects/jason/pickle/salvage1.pkl")

    print('salvaged, we can start again with /infodev1/rep/projects/jason/pickle/lstmsalvage1.pkl')

def run_one_step(computer, channelmanager, optimizer, binary_criterion):
    computer.train()
    optimizer.zero_grad()
    input, target, loss_type, states_tuple = next(channelmanager)
    target=Variable(target.squeeze(1).cuda())
    input=Variable(input).cuda()
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
    target = Variable(target.cuda())
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
    with open(logfile, 'a') as handle:
        handle.write(string)
    print(string)

def train(computer, optimizer, real_criterion, binary_criterion,
          train, valid, starting_epoch, total_epochs, starting_iter, iter_per_epoch, savestr, logfile=True):
    print_interval = 100
    val_interval = 1000
    save_interval = 1000
    target_dim = None
    rldmax_len = 500
    val_batch = 500
    running_loss_deque = deque(maxlen=rldmax_len)

    # erase the logfile
    if logfile:
        logfile=str(datetime.datetime.now())


    for epoch in range(starting_epoch, total_epochs):
        # all these are batches
        for i in range(starting_iter, iter_per_epoch):
            train_step_loss = run_one_step(computer, train, optimizer, binary_criterion)
            if train_step_loss is not None:
                printloss = float(train_step_loss[0])
            else:
                raise ValueError("What is happening?")
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
                    assert(printloss==printloss)
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

# def train_obsolete(computer, optimizer, real_criterion, binary_criterion,
#           train, valid_dl, starting_epoch, total_epochs, starting_iter, iter_per_epoch, logfile=False):
#     valid_iterator=iter(valid_dl)
#     print_interval=10
#     val_interval=200
#     save_interval=800
#     target_dim=None
#     rldmax_len=50
#     val_batch=100
#     running_loss_deque=deque(maxlen=rldmax_len)
#     if logfile:
#         open(logfile, 'w').close()
#
#     for epoch in range(starting_epoch, total_epochs):
#         for i, (input, target, loss_type) in enumerate(train):
#             i=starting_iter+i
#             if target_dim is None:
#                 target_dim=target.shape[2]
#
#             if i < iter_per_epoch:
#                 train_story_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
#                                                    real_criterion, binary_criterion)
#                 if train_story_loss is not None:
#                     printloss=float(train_story_loss[0])
#                 else:
#                     raise ValueError("Why would story loss be None?")
#                 running_loss_deque.appendleft(printloss)
#                 if i % print_interval == 0:
#                     running_loss=np.mean(running_loss_deque)
#                     if logfile:
#                         with open(logfile, 'a') as handle:
#                             handle.write("learning.   count: %4d, training loss: %.10f \n" %
#                                          (i, printloss))
#                     print("learning.   count: %4d, training loss: %.10f" %
#                           (i, printloss))
#                     if i!=0:
#                         print("count: %4d, running loss: %.10f" % (i, running_loss))
#
#                 if i % val_interval == 0:
#                     printloss=0
#                     for _ in range(val_batch):
#                         # we should consider running validation multiple times and average. TODO
#                         try:
#                             (input,target,loss_type)=next(valid_iterator)
#                         except StopIteration:
#                             valid_iterator=iter(valid_dl)
#                             (input,target,loss_type)=next(valid_iterator)
#
#                         val_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
#                                                        real_criterion, binary_criterion, validate=True)
#                         if val_loss is not None:
#                             printloss += float(val_loss[0])
#                         else:
#                             raise ValueError ("Investigate this")
#                     printloss=printloss/val_batch
#                     if logfile:
#                         with open(logfile, 'a') as handle:
#                             handle.write("validation. count: %4d, val loss     : %.10f \n" %
#                                              (i, printloss))
#                     print("validation. count: %4d, val loss: %.10f" %
#                           (i, printloss))
#
#                 if i % save_interval == 0:
#                     save_model(computer, optimizer, epoch, i)
#                     print("model saved for epoch", epoch, "input", i)
#             else:
#                 break

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
        if logfile:
            logprint(logfile,"validation. count: %4d, val loss     : %.10f" %
                             (i, printloss))
        print("validation. count: %4d, loss: %.10f" %
              (i, printloss))
    print("loss:",np.mean(val_losses))


def validationonly():
    '''
    :return:
    '''

    lr = 1e-2
    optim = None
    logfile = "vallog.txt"

    num_workers = 8
    ig = InputGenD()
    # multiprocessing disabled, because socket request seems unstable.
    # performance should not be too bad?
    trainds, validds = train_valid_split(ig, split_fold=10)
    validdl = DataLoader(dataset=validds,num_workers=num_workers, batch_size=1)
    print("Using", num_workers, "workers for validation set")
    # testing whether this LSTM works is basically a question whether
    lstm = ChannelLSTM()

    # load model:
    print("loading model")
    lstm, optim, starting_epoch, starting_iteration = load_model(lstm, optim, 0, 0)

    lstm = lstm.cuda()
    if optim is None:
        optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)
    else:
        # print('use Adadelta optimizer with learning rate ', lr)
        # optimizer = torch.optim.Adadelta(computer.parameters(), lr=lr)
        optimizer = optim

    real_criterion = nn.SmoothL1Loss()
    binary_criterion = nn.BCEWithLogitsLoss()

    traindl=None
    total_epochs=None
    iter_per_epoch=None

    # starting with the epoch after the loaded one
    valid(lstm, optimizer, real_criterion, binary_criterion,
          traindl, validdl, int(starting_epoch), total_epochs,int(starting_iteration), iter_per_epoch, logfile)


def main(load=False,savestr="lstm"):
    """
    11/29
    lr=1e-2
    bottom loss 0.0004
    val loss 0.002~0.001 afterwards, which is comparable to DNC
    There are signs of sparse output, because sometimes the prediction is exactly correct.
    """
    total_epochs = 10
    iter_per_epoch = 100000
    lr = 1e-4
    optim = None
    starting_epoch = 0
    starting_iteration= 0
    logstring = str(datetime.datetime.now().time())
    logstring.replace(" ", "_")
    logfile = "log/"+savestr+"_"+logstring+".txt"
    param_bs=16

    num_workers = 16
    lstm=ChannelLSTM()

    ig = InputGenF(death_fold=0)
    trainds = ig.get_train()
    validds = ig.get_valid()
    testds = ig.get_test()
    traindl = DataLoader(dataset=trainds, batch_size=1, num_workers=num_workers)
    validdl = DataLoader(dataset=validds, batch_size=1, num_workers=num_workers)
    traindl = ChannelManager(traindl, param_bs, model=lstm)
    validdl = ChannelManager(validdl, param_bs, model=lstm)

    print("Using", num_workers, "workers for training set")
    # testing whether this LSTM works is basically a question whether

    # load model:
    if load:
        print("loading model")
        lstm, optim, starting_epoch, starting_iteration = load_model(lstm, optim, starting_epoch, starting_iteration, savestr)

    lstm = lstm.cuda()
    if optim is None:
        optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)
    else:
        # print('use Adadelta optimizer with learning rate ', lr)
        # optimizer = torch.optim.Adadelta(computer.parameters(), lr=lr)
        optimizer = optim

    real_criterion = nn.SmoothL1Loss()
    binary_criterion = nn.BCEWithLogitsLoss()

    # starting with the epoch after the loaded one

    train(lstm, optimizer, real_criterion, binary_criterion,
          traindl, validdl, int(starting_epoch), total_epochs,
          int(starting_iteration), iter_per_epoch, savestr, logfile)



if __name__ == "__main__":
    # main(load=True
    main(load=True,savestr="cnlstm")

    '''
    Loss is around 0.004
    Validation set was exhausted. We should make a wrapper and reuse it.w
    Hidden size 128, layer size 16
    '''