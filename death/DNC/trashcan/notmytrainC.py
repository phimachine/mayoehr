import pandas as pd
from dnc import DNC
import torch
import numpy
import pdb
from pathlib import Path
import os
from os.path import abspath
from death.post.inputgen_planC import InputGen
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import archi.param as param
import torch.nn as nn


class DNCwrapper(DNC):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 rnn_type='lstm',
                 num_layers=1,
                 num_hidden_layers=2,
                 bias=True,
                 batch_first=True,
                 dropout=0,
                 bidirectional=False,
                 nr_cells=5,
                 read_heads=2,
                 cell_size=10,
                 nonlinearity='tanh',
                 gpu_id=-1,
                 independent_linears=False,
                 share_memory=True,
                 debug=False,
                 clip=20):
        super(DNCwrapper, self).__init__(input_size,
                                         hidden_size,
                                         rnn_type=rnn_type,
                                         num_layers=num_layers,
                                         num_hidden_layers=num_hidden_layers,
                                         bias=bias,
                                         batch_first=batch_first,
                                         dropout=dropout,
                                         bidirectional=bidirectional,
                                         nr_cells=nr_cells,
                                         read_heads=read_heads,
                                         cell_size=cell_size,
                                         nonlinearity=nonlinearity,
                                         gpu_id=gpu_id,
                                         independent_linears=independent_linears,
                                         share_memory=share_memory,
                                         debug=debug,
                                         clip=clip)
        self.og = torch.nn.Linear(self.input_size, output_size, bias=True)

    def forward(self, input, hx=(None, None, None), reset_experience=False, pass_through_memory=True):
        if self.debug:
            outputs, (controller_hidden, mem_hidden, read_vectors), viz = \
                super(DNCwrapper, self).forward(input, hx=(None, None, None), reset_experience=False,
                                                pass_through_memory=True)
            outputs = self.og(outputs)
            return outputs, (controller_hidden, mem_hidden, read_vectors), viz
        else:
            outputs, (controller_hidden, mem_hidden, read_vectors) = \
                super(DNCwrapper, self).forward(input, hx=(None, None, None), reset_experience=False,
                                                pass_through_memory=True)
            outputs = self.og(outputs)
            return outputs, (controller_hidden, mem_hidden, read_vectors)


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def save_model(net, optim, epoch, i):
    epoch = int(epoch)
    task_dir = os.path.dirname(abspath(__file__))
    pickle_file = Path(task_dir).joinpath("saves/DNCfull_" + str(epoch) + "_" + str(i) + ".pkl")
    pickle_file = pickle_file.open('wb')
    torch.save((net, optim, epoch), pickle_file)


def load_model(computer):
    task_dir = os.path.dirname(abspath(__file__))
    save_dir = Path(task_dir) / "saves"
    highestepoch = -1
    for child in save_dir.iterdir():
        epoch = str(child).split("_")[2].split('.')[0]
        epoch = int(epoch)
        # some files are open but not written to yet.
        if epoch > highestepoch and child.stat().st_size > 2048:
            highestepoch = epoch
    if highestepoch == -1:
        return computer, None, -1
    pickle_file = Path(task_dir).joinpath("saves/DNCfull_" + str(highestepoch) + ".pkl")
    print("loading model at ", pickle_file)
    pickle_file = pickle_file.open('rb')
    model, optim, epoch = torch.load(pickle_file)

    print('Loaded model at epoch ', highestepoch)

    for child in save_dir.iterdir():
        epoch = str(child).split("_")[2].split('.')[0]
        if int(epoch) != highestepoch:
            os.remove(child)
    print('Removed incomplete save file and all else.')

    return model, optim, epoch


def run_one_patient(computer, input, target, optimizer, loss_type, real_criterion,
                    binary_criterion, memory_hidden, validate=False, debug=False):


    with torch.no_grad if validate else dummy_context_mgr():
        if debug:
            patient_output, (_, memory_hidden, _), v = \
                computer(input, (None, memory_hidden, None), reset_experience=True, pass_through_memory=True)
        else:
            patient_output, (_, memory_hidden, _) = \
                computer(input, (None, memory_hidden, None), reset_experience=True, pass_through_memory=True)
        assert not (patient_output != patient_output).any()

        # patient_output: (batch_size 1, time_length, output_dim ~4000)
        time_to_event_output = patient_output[:, :, 0]
        cause_of_death_output = patient_output[:, :, 1:]
        time_to_event_target = target[:, :, 0]
        cause_of_death_target = target[:, :, 1:]

        if loss_type[0] == 0:
            # in record
            toe_loss = real_criterion(time_to_event_output, time_to_event_target)
            cod_loss = binary_criterion(cause_of_death_output, cause_of_death_target)
            patient_loss = toe_loss + cod_loss
        else:
            # not in record
            # be careful with the sign, penalize when and only when positive
            underestimation = time_to_event_target - time_to_event_output
            underestimation = nn.ReLU(underestimation)
            toe_loss = real_criterion(underestimation, 0)
            cod_loss = binary_criterion(cause_of_death_output, cause_of_death_target)
            patient_loss = toe_loss + cod_loss

        # patient_loss.requires_grad=True

        if not validate:
            patient_loss.backward()
            optimizer.step()

    return patient_loss, memory_hidden


def train(computer, optimizer, real_criterion, binary_criterion,
          igdl, starting_epoch, total_epochs):
    device = torch.cuda.device("cuda:0" if torch.cuda.is_available() else "cpu")

    memory_hidden = None

    for epoch in range(starting_epoch, total_epochs):

        # running_loss = 0

        for i, (input, target, loss_type) in enumerate(igdl):
            input, target, loss_type= Variable(input,requires_grad=True).cuda(), Variable(target).cuda(), Variable(loss_type)

            patient_loss, memory_hidden = run_one_patient(computer, input, target, optimizer, loss_type,
                                                          real_criterion, binary_criterion, memory_hidden)
            if i % 10 == 0:
                print("learning. count: %4d, training loss: %.4f" %
                      (i, patient_loss[0]))

            # This line might be holding an object that causes CPU memory leak.
            # I didn't need a running loss anyway. unbelievable.
            # running_loss += patient_loss

            # TODO No validation support for now.
            # val_freq = 16
            # if batch % val_freq == val_freq - 1:
            #     print('summary.  epoch: %4d, batch number: %4d, running loss: %.4f' %
            #           (epoch, batch, running_loss / val_freq))
            #     running_loss = 0
            #     # also test the model
            #     val_loss = run_one_story(computer, optimizer, story_length, batch_size, pgd, validate=False)
            #     print('validate. epoch: %4d, batch number: %4d, validation loss: %.4f' %
            #           (epoch, batch, val_loss))

            if i % 100 == 99:
                save_model(computer, optimizer, epoch, i)
                print("model saved for epoch ", epoch, "-", i)


if __name__ == "__main__":
    total_epochs = 1000
    lr = 1e-5
    optim = None
    starting_epoch = -1

    ig = InputGen()
    igdl = DataLoader(dataset=ig, batch_size=1, shuffle=True, num_workers=16)

    computer = DNCwrapper(
        input_size=param.x,
        output_size=param.v_t,
        hidden_size=128,
        rnn_type='lstm',
        num_layers=8,
        nr_cells=100,
        cell_size=32,
        read_heads=4,
        batch_first=True,
        gpu_id=0
    )
    computer=nn.DataParallel(computer)


    # if load model
    # computer, optim, starting_epoch = load_model(computer)

    computer = computer.cuda()
    if optim is None:
        optimizer = torch.optim.Adam(computer.parameters(), lr=lr)
    else:
        print('use Adadelta optimizer with learning rate ', lr)
        optimizer = torch.optim.Adadelta(computer.parameters(), lr=lr)

    real_criterion = nn.SmoothL1Loss()
    binary_criterion = nn.BCEWithLogitsLoss()

    # starting with the epoch after the loaded one

    train(computer, optimizer, real_criterion, binary_criterion,
          igdl, int(starting_epoch) + 1, total_epochs)

'''
It runs, but it's also unbearably slow. Very slow.
'''