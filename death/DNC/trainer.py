import pandas as pd
from archi.computer import Computer
import torch
import numpy
import pdb
from pathlib import Path
import os
from os.path import abspath
from death.post.inputgen_planA import InputGen
from torch.utils.data import DataLoader

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


def run_one_patient(computer, optimizer, criterion, input, target, validate=False):
    '''

    :param computer:
    :param optimizer:
    :param input:
    :param target:
    :param validate: right now, validation is not supported, must always be false.
    :return:
    '''

    input = torch.Tensor(input).cuda()
    target = torch.Tensor(target).cuda()

    # we have no critical index, becuase critical index are those timesteps that
    # DNC is required to produce outputs. This is not the case for our project.
    # criterion does not need to be initiated here, because we are not using a mask

    time_length = input.size()[1]
    with torch.no_grad if validate else dummy_context_mgr():
        patient_output = torch.Tensor(1, time_length, param.output_dim)
        computer.new_sequence_reset()
        for timestep in range(time_length):
            # first colon is always size 1
            feeding = input[:, timestep, :]
            output = computer(feeding)
            assert not torch.isnan(output).any()
            patient_output[1, timestep, :] = output

        story_loss = criterion(patient_output, target)
        ## pass criterion on.
        if not validate:
            # I chose to backward a derivative only after a whole story has been taken in
            # This should lead to a more stable, but initially slower convergence.
            story_loss.backward()
            optimizer.step()

    return story_loss


def train(computer, optimizer, criterion, igdl, starting_epoch, total_epochs):
    for epoch in range(starting_epoch, total_epochs):

        running_loss = 0

        for i, (input, target) in igdl:

            train_story_loss = run_one_patient(computer, optimizer, criterion, input, target)
            if i % 100 == 0:
                print("learning. count: %4d, training loss: %.4f" %
                      (i, train_story_loss.item()))
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

    # starting with the epoch after the loaded one
    train(computer, optimizer, criterion, igdl, int(starting_epoch) + 1, total_epochs)
