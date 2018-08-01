"""
D2 is a modification oif the DNC model that resets the memory at every new story.
This is how people usually implement DNC.
Use the parameter to control whether experience resets.
"""

import pandas as pd
from dnc import SDNC
import torch
import numpy as np
import pdb
from pathlib import Path
import os
from os.path import abspath
from death.post.inputgen_planD import InputGenD, train_valid_split
from torch.utils.data import DataLoader
import torch.nn as nn
from death.DNC.frankenstein2 import Frankenstein as DNC
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


def run_one_patient(computer, input, target, target_dim, optimizer, loss_type, real_criterion,
                    binary_criterion, validate=False):
    global global_exception_counter
    global i
    patient_loss = None
    if debug:
        if (input != input).any():
            raise ValueError("NA in input")
        if (target != target).any():
            raise ValueError("NA in target")
    try:
        (controller_hidden, memory, read_vectors, reset_experience) = (None, None, None, True)
        optimizer.zero_grad()
        input = Variable(torch.Tensor(input).cuda())
        target = Variable(torch.Tensor(target).cuda())

        # we have no critical index, becuase critical index are those timesteps that
        # DNC is required to produce outputs. This is not the case for our project.
        # criterion does not need to be reinitiated for every story, because we are not using a mask

        time_length = input.size()[1]
        # with torch.no_grad if validate else dummy_context_mgr():
        patient_output = Variable(torch.Tensor(1, time_length, target_dim)).cuda()
        output, (controller_hidden, memory, read_vectors) = \
            computer(input, (controller_hidden, memory, read_vectors, reset_experience))
        assert not (output != output).any()

        # for timestep in range(time_length):
        #     # first colon is always size 1
        #     feeding = input[:, timestep, :]
        #
        #     output, (controller_hidden, memory, read_vectors), debug_memory = \
        #         computer(feeding, (controller_hidden, memory, read_vectors, reset_experience))
        #     reset_experience = False
        #
        #     # output = computer(feeding)
        #     assert not (output != output).any()
        #     patient_output[0, timestep, :] = output

        # patient_output: (batch_size 1, time_length, output_dim ~4000)
        time_to_event_output = patient_output[:, :, 0]
        cause_of_death_output = patient_output[:, :, 1:]
        time_to_event_target = target[:, :, 0]
        cause_of_death_target = target[:, :, 1:]

        # this block will not work for batch input,
        # you should modify it so that the loss evaluation is not determined by logic but function.
        # def toe_loss_calc(real_criterion,time_to_event_output,time_to_event_target, patient_length):
        #
        # if loss_type[0] == 0:
        #     # in record
        #     toe_loss = real_criterion(time_to_event_output, time_to_event_target)
        #     cod_loss = binary_criterion(cause_of_death_output, cause_of_death_target)
        #     patient_loss = toe_loss/100 + cod_loss
        # else:
        #     # not in record
        #     # be careful with the sign, penalize when and only when positive
        #     underestimation = time_to_event_target - time_to_event_output
        #     underestimation = nn.functional.relu(underestimation)
        #     toe_loss = real_criterion(underestimation, torch.zeros_like(underestimation).cuda())
        #     cod_loss = binary_criterion(cause_of_death_output, cause_of_death_target)
        #     patient_loss = toe_loss/100 + cod_loss
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
            save_model(computer, optimizer, epoch=0, iteration=np.random.randint(0, 1000), savestr="NA")
            task_dir = os.path.dirname(abspath(__file__))
            save_dir = Path(task_dir) / "saves" / "probleminput.pkl"
            with save_dir.open('wb') as fhand:
                pickle.dump(input, fhand)
            raise ValueError("Global exception counter reached 10. Likely the model has nan in weights")
        else:
            print("we are at", i)
            pass

    return patient_loss


def train(computer, optimizer, real_criterion, binary_criterion,
          train, valid_iterator, starting_epoch, total_epochs, starting_iter, iter_per_epoch, savestr, logfile=False):
    global global_exception_counter

    print_interval = 10
    val_interval = 50
    save_interval = 300
    target_dim = None
    rldmax_len = 50
    val_batch = 10
    running_loss_deque = deque(maxlen=rldmax_len)
    if logfile:
        open(logfile, 'w').close()
    global i

    for epoch in range(starting_epoch, total_epochs):
        for i, (input, target, loss_type) in enumerate(train):
            i = starting_iter + i
            if target_dim is None:
                target_dim = target.shape[2]

            if i < iter_per_epoch:
                train_story_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                                   real_criterion, binary_criterion)
                if train_story_loss is not None:
                    printloss = float(train_story_loss[0])
                else:
                    printloss = 10000
                del input, target, loss_type
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

                if i % val_interval == 0:
                    for _ in range(val_batch):
                        printloss = 0
                        (input, target, loss_type) = next(valid_iterator)
                        val_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                                   real_criterion, binary_criterion, validate=True)
                        if val_loss is not None:
                            printloss += float(val_loss[0])
                    printloss = printloss / val_batch
                    if logfile:
                        with open(logfile, 'a') as handle:
                            handle.write("validation. count: %4d, val loss     : %.10f \n" %
                                         (i, printloss))
                    print("validation. count: %4d, training loss: %.10f" %
                          (i, printloss))

                if i % save_interval == 0:
                    save_model(computer, optimizer, epoch, i, savestr)
                    print("model saved for epoch", epoch, "input", i)
            else:
                break


def forevermain(load=False, lr=1e-3, savestr="", reset=True, palette=False):
    print("Will run main() forever in a loop.")
    while True:
        try:
            main(load, lr, savestr, reset, palette)
        except ValueError:
            traceback.print_exc()


class NotMySam(SDNC):
    def __init__(
            self,
            input_size,
            hidden_size,
            last_output_size,
            rnn_type='lstm',
            num_layers=1,
            num_hidden_layers=2,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
            nr_cells=5000,
            sparse_reads=4,
            temporal_reads=4,
            read_heads=4,
            cell_size=10,
            nonlinearity='tanh',
            gpu_id=-1,
            independent_linears=False,
            share_memory=True,
            debug=False,
            clip=20):
        super(NotMySam, self).__init__(input_size,
                                       hidden_size,
                                       rnn_type=rnn_type,
                                       num_layers=num_layers,
                                       num_hidden_layers=num_hidden_layers,
                                       bias=bias,
                                       batch_first=batch_first,
                                       dropout=dropout,
                                       bidirectional=bidirectional,
                                       nr_cells=nr_cells,
                                       sparse_reads=sparse_reads,
                                       temporal_reads=temporal_reads,
                                       read_heads=read_heads,
                                       cell_size=cell_size,
                                       nonlinearity=nonlinearity,
                                       gpu_id=gpu_id,
                                       independent_linears=independent_linears,
                                       share_memory=share_memory,
                                       debug=debug,
                                       clip=clip)

        self.last_output_size=last_output_size
        self.last_output = nn.Linear(self.input_size, self.last_output_size)
    
    def forward(self, input, hx=(None, None, None), reset_experience=False, pass_through_memory=True):
        if self.debug:
            outputs, (controller_hidden, mem_hidden, read_vectors), viz=\
                super(NotMySam, self).forward(input, hx=(None, None, None), reset_experience=False, pass_through_memory=True)
            outputs = self.last_output(outputs)

            return outputs, (controller_hidden,mem_hidden,read_vectors), viz
        else:
            outputs, (controller_hidden, mem_hidden, read_vectors)=\
                super(NotMySam, self).forward(input, hx=(None, None, None), reset_experience=False, pass_through_memory=True)
            outputs = self.last_output(outputs)

            return outputs, (controller_hidden,mem_hidden,read_vectors)

def main(load=False, lr=1e-3, savestr="", reset=True, palette=False):
    total_epochs = 10
    iter_per_epoch = 100000
    lr = lr
    optim = None
    starting_epoch = 0
    starting_iteration = 0
    logfile = "log.txt"

    num_workers = 3
    ig = InputGenD()
    trainds, validds = train_valid_split(ig, split_fold=10)
    traindl = DataLoader(dataset=trainds, batch_size=1, num_workers=num_workers)
    validdl = DataLoader(dataset=validds, batch_size=1)
    print("Using", num_workers, "workers for training set")
    computer = NotMySam(
        input_size=47764,
        hidden_size=128,
        last_output_size=3620,
        rnn_type='lstm',
        num_layers=4,
        nr_cells=100,
        cell_size=32,
        read_heads=4,
        sparse_reads=4,
        batch_first=True,
        gpu_id=0
    )
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
    binary_criterion = nn.BCEWithLogitsLoss(size_average=False)

    # starting with the epoch after the loaded one

    train(computer, optimizer, real_criterion, binary_criterion,
          traindl, iter(validdl), int(starting_epoch), total_epochs, int(starting_iteration), iter_per_epoch, savestr,
          logfile)


if __name__ == "__main__":
    main()
