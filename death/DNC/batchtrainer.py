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

param_x = 47764
param_h = 128
param_L = 16
param_v_t = 3620
param_W = 32
param_R = 8
param_N = 512
param_bs = 4
param_reset = True


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
#
# def run_one_patient(computer, input, target, reset_flag, states_tuple, target_dim, optimizer, loss_type, real_criterion,
#                     binary_criterion, validate=False):
#     global global_exception_counter
#     global i
#     patient_loss = None
#     if debug:
#         if (input != input).any():
#             raise ValueError("NA in input")
#         if (target != target).any():
#             raise ValueError("NA in target")
#     try:
#         optimizer.zero_grad()
#         input = Variable(input.cuda())
#         target = Variable(target.cuda())
#
#         # process the state tuples and reset if needed TODO
#
#         output = computer(input, states_tuple)
#         time_to_event_output = output[:, :, 0]
#         cause_of_death_output = output[:, :, 1:]
#         time_to_event_target = target[:, :, 0]
#         cause_of_death_target = target[:, :, 1:]
#
#         if debug:
#             assert not (output != output).any()
#         patient_loss = binary_criterion(cause_of_death_output, cause_of_death_target)
#
#         if not validate:
#             patient_loss.backward()
#             optimizer.step()
#
#         if global_exception_counter > -1:
#             global_exception_counter -= 1
#
#     except ValueError:
#         traceback.print_exc()
#         print("Value Error reached")
#         print(datetime.datetime.now().time())
#         global_exception_counter += 1
#         if global_exception_counter == 10:
#             save_model(computer, optimizer, epoch=0, iteration=np.random.randint(0, 1000), savestr="NA")
#             task_dir = os.path.dirname(abspath(__file__))
#             save_dir = Path(task_dir) / "saves" / "probleminput.pkl"
#             with save_dir.open('wb') as fhand:
#                 pickle.dump(input, fhand)
#             raise ValueError("Global exception counter reached 10. Likely the model has nan in weights")
#         else:
#             print("we are at", i)
#             pass
#
#     return patient_loss, states_tuple

def run_one_step(computer, channelmanager, optimizer, binary_criterion):
    optimizer.zero_grad()
    input, target, loss_type, states_tuple = next(channelmanager)
    input=Variable(input).cuda()
    target=Variable(target).cuda()
    loss_type=Variable(loss_type).cuda()
    computer.assign_states_tuple(states_tuple)
    output, states_tuple = computer(input)
    bc.push_states(states_tuple)
    loss=binary_criterion(cause_of_death_output, cause_of_death_target)
    loss.backward()
    optimizer.step()
    return loss


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
    print_interval = 10
    val_interval = 50
    save_interval = 300
    target_dim = None
    rldmax_len = 50
    val_batch = 10
    running_loss_deque = deque(maxlen=rldmax_len)

    # erase the logfile
    if logfile:
        open(logfile, 'w').close()
    global i

    for epoch in range(starting_epoch, total_epochs):
        # all these are batches
        for i in range(starting_iter,iter_per_epoch):
            train_story_loss = run_one_step(computer, train, optimizer, binary_criterion)
            if train_story_loss is not None:
                printloss = float(train_story_loss[0])
            else:
                printloss = 10000
            # computer.new_sequence_reset()
            running_loss_deque.appendleft(printloss)
            if i % print_interval == 0:
                running_loss = np.mean(running_loss_deque)
                if logfile:
                    with open(logfile, 'a') as handle:
                        handle.write("learning.   count: %4d, training loss: %.10f \n" %
                                     (i, printloss))
                print("learning.   count: %4d, training loss: %.10f" %
                      (i, printloss))
                if i != 0:
                    print("count: %4d, running loss: %.10f" % (i, running_loss))
            #
            # if i % val_interval == 0:
            #     for _ in range(val_batch):
            #         printloss = 0
            #         (input, target, loss_type) = next(valid_iterator)
            #         val_loss, valid_loss_tuple = run_one_patient(computer, input, target, reset_flag,
            #                                                      valid_states_tuple,
            #                                                      target_dim, optimizer, loss_type,
            #                                                      real_criterion, binary_criterion, validate=True)
            #         if val_loss is not None:
            #             printloss += float(val_loss[0])
            #     printloss = printloss / val_batch
            #     if logfile:
            #         with open(logfile, 'a') as handle:
            #             handle.write("validation. count: %4d, val loss     : %.10f \n" %
            #                          (i, printloss))
            #     print("validation. count: %4d, training loss: %.10f" %
            #           (i, printloss))

            if i % save_interval == 0:
                save_model(computer, optimizer, epoch, i, savestr)
                print("model saved for epoch", epoch, "input", i)


def forevermain(load=False, lr=1e-3, savestr="", reset=True, palette=False):
    print("Will run main() forever in a loop.")
    while True:
        try:
            main(load, lr, savestr, reset, palette)
        except ValueError:
            traceback.print_exc()


def main(load=False, lr=1e-3, savestr="", reset=True, palette=False):
    total_epochs = 10
    iter_per_epoch = 100000
    lr = lr
    optim = None
    starting_epoch = 0
    starting_iteration = 0
    logfile = "log.txt"
    num_workers=8

    print("Using", num_workers, "workers for training set")
    computer = DNC(x=param_x,
                   h=param_h,
                   L=param_L,
                   v_t=param_v_t,
                   W=param_W,
                   R=param_R,
                   N=param_N,
                   bs=param_bs)

    ig = InputGenD(load_pickle=True, verbose=False)
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


if __name__ == "__main__":
    main()
