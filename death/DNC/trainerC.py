import pandas as pd
from archi.computer import Computer
import torch
import numpy
import pdb
from pathlib import Path
import os
from os.path import abspath
from death.post.inputgen_planC import InputGen
from torch.utils.data import DataLoader
import torch.nn as nn
import archi.param as param
from torch.autograd import Variable

batch_size = 1


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

def save_model(net, optim, epoch):
    epoch = int(epoch)
    task_dir = os.path.dirname(abspath(__file__))
    pickle_file = Path(task_dir).joinpath("saves/DNCfull_" + str(epoch) + ".pkl")
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
                    binary_criterion, validate=False):

    input = Variable(torch.Tensor(input).cuda())
    target = Variable(torch.Tensor(target).cuda())

    # we have no critical index, becuase critical index are those timesteps that
    # DNC is required to produce outputs. This is not the case for our project.
    # criterion does not need to be reinitiated for every story, because we are not using a mask

    time_length = input.size()[1]
    with torch.no_grad if validate else dummy_context_mgr():
        patient_output = Variable(torch.Tensor(1, time_length, param.v_t)).cuda()
        computer.new_sequence_reset()
        for timestep in range(time_length):
            # first colon is always size 1
            feeding = input[:, timestep, :]
            output = computer(feeding)
            assert not (output!=output).any()
            patient_output[0, timestep, :] = output.data

        # patient_output: (batch_size 1, time_length, output_dim ~4000)
        time_to_event_output=patient_output[:,:,0]
        cause_of_death_output=patient_output[:,:,1:]
        time_to_event_target=target[:,:,0]
        cause_of_death_target=target[:,:,1:]

        patient_loss=None

        # this block will not work for batch input,
        # you should modify it so that the loss evaluation is not determined by logic but function.
        if loss_type[0]==0:
            # in record
            toe_loss = real_criterion(time_to_event_output,time_to_event_target)
            cod_loss = binary_criterion(cause_of_death_output,cause_of_death_target)
            patient_loss=toe_loss+cod_loss
        else:
            # not in record
            # be careful with the sign, penalize when and only when positive
            underestimation = time_to_event_target-time_to_event_output
            underestimation = nn.ReLU(underestimation)
            toe_loss = real_criterion(underestimation,0)
            cod_loss = binary_criterion(cause_of_death_output,cause_of_death_target)
            patient_loss=toe_loss+cod_loss

        patient_loss.requires_grad=True

        if not validate:
            patient_loss.backward()
            optimizer.step()

    return patient_loss


def train(computer, optimizer, real_criterion, binary_criterion,
          igdl, starting_epoch, total_epochs):

    for epoch in range(starting_epoch, total_epochs):

        running_loss = 0

        for i, (input, target, loss_type) in enumerate(igdl):

            train_story_loss = run_one_patient(computer, input, target, optimizer, loss_type,
                                               real_criterion,binary_criterion)
            if i % 100 == 0:
                print("learning. count: %4d, training loss: %.4f" %
                      (i, train_story_loss[0]))
            running_loss += train_story_loss
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

        save_model(computer, optimizer, epoch)
        print("model saved for epoch ", epoch)


if __name__=="__main__":
    total_epochs = 1000
    lr = 1e-5
    optim = None
    starting_epoch = -1

    ig=InputGen()
    igdl=DataLoader(dataset=ig,batch_size=1,shuffle=False,num_workers=16)

    computer = Computer()

    # if load model
    # computer, optim, starting_epoch = load_model(computer)

    computer = computer.cuda()
    if optim is None:
        optimizer = torch.optim.Adam(computer.parameters(), lr=lr)
    else:
        print('use Adadelta optimizer with learning rate ', lr)
        optimizer = torch.optim.Adadelta(computer.parameters(), lr=lr)

    real_criterion=nn.SmoothL1Loss()
    binary_criterion=nn.BCEWithLogitsLoss()

    # starting with the epoch after the loaded one

    train(computer, optimizer, real_criterion, binary_criterion,
          igdl, int(starting_epoch) + 1, total_epochs)
