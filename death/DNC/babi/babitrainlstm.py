# I decide to run my model on babi again to see if the convergen ce problem is with my model or dataset

from death.DNC.frankenstein import Frankenstein as DNC
import torch
import numpy
import death.DNC.archi.param as param
import pdb
from pathlib import Path
import os
from os.path import abspath
import gc
import time
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists
import pickle
import numpy as np
from shutil import rmtree
import os
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists
import subprocess
import death.DNC.archi.param as param
from threading import Thread
import time
from death.DNC.babi.babigen import PreGenData
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.modules import LSTM


# task 10 of babi

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


def save_model_old(net, optim, epoch):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    task_dir = os.path.dirname(abspath(__file__))
    print(task_dir)
    pickle_file = Path("../saves/DNC_" + str(epoch) + ".pkl")
    pickle_file = pickle_file.open('wb')

    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        pickle_file)


def load_model_old(net):
    task_dir = os.path.dirname(abspath(__file__))
    save_dir = Path(task_dir).parent / "saves"
    highestepoch = -1
    for child in save_dir.iterdir():
        epoch = str(child).split("_")[1].split('.')[0]
        # some files are open but not written to yet.
        if int(epoch) > highestepoch and child.stat().st_size > 2048:
            highestepoch = int(epoch)
    pickle_file = Path("../saves/DNC_" + str(highestepoch) + ".pkl")
    pickle_file = pickle_file.open('rb')
    ret = torch.load(pickle_file)

    net.load_state_dict(ret['state_dict'])
    print('Loaded model at epoch ', highestepoch)

    for child in save_dir.iterdir():
        epoch = str(child).split("_")[1].split('.')[0]
        if int(epoch) != highestepoch:
            os.remove(child)
    print('Removed incomplete save file and all else.')

    return ret['epoch'], ret['optimizer']


def run_one_story(computer, optimizer, story_length, batch_size, pgd, input_dim, validate=False):
    optimizer.zero_grad()
    # to promote code reuse
    if not validate:
        input_data, target_output, critical_index = pgd.get_train()
    else:
        input_data, target_output, critical_index = pgd.get_validate()

    input_data = Variable(torch.Tensor(input_data).cuda())
    target_output = Variable(torch.Tensor(target_output).cuda())
    stairs = torch.Tensor(numpy.arange(0, batch_size * story_length, story_length))
    critical_index = critical_index + stairs.unsqueeze(1)
    critical_index = critical_index.view(-1)
    critical_index = critical_index.long().cuda()

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad if validate else dummy_context_mgr():

        story_output=computer(input_data)
        #
        # story_output = Variable(torch.Tensor(batch_size, story_length, input_dim).cuda())
        # # a single story
        # for timestep in range(story_length):
        #     # feed the batch into the machine
        #     # Expected input dimension: (150, 27)
        #     # output: (150,27)
        #     batch_input_of_same_timestep = input_data[:, timestep, :]
        #
        #     # usually batch does not interfere with each other's logic
        #     batch_output = computer(batch_input_of_same_timestep)
        #     if (batch_output!=batch_output).any():
        #         pdb.set_trace()
        #         raise ValueError("nan is found in the batch output.")
        #     story_output[:, timestep, :] = batch_output

        target_output = target_output.view(-1)
        story_output = story_output.view(-1, input_dim)
        story_output = story_output[critical_index, :]
        target_output = target_output[critical_index].long()

        story_loss = criterion(story_output, target_output)
        if not validate:
            # I chose to backward a derivative only after a whole story has been taken in
            # This should lead to a more stable, but initially slower convergence.
            story_loss.backward()
            optimizer.step()

    return story_loss


def train(computer, optimizer, story_length, batch_size, pgd, input_dim, starting_epoch, epochs_count, epoch_batches_count):
    for epoch in range(starting_epoch, epochs_count):

        running_loss = 0

        for batch in range(epoch_batches_count):

            train_story_loss = run_one_story(computer, optimizer, story_length, batch_size, pgd, input_dim)
            print("learning. epoch: %4d, batch number: %4d, training loss: %.4f" %
                  (epoch, batch, train_story_loss[0]))
            running_loss += float(train_story_loss[0])
            val_freq = 16
            if batch % val_freq == val_freq - 1:
                print('summary.  epoch: %4d, batch number: %4d, running loss: %.4f' %
                      (epoch, batch, running_loss / val_freq))
                running_loss = 0
                # also test the model
                val_loss = run_one_story(computer, optimizer, story_length, batch_size, pgd, input_dim, validate=False)
                print('validate. epoch: %4d, batch number: %4d, validation loss: %.4f' %
                      (epoch, batch, float(val_loss[0])))

        save_model(computer, optimizer, epoch)
        print("model saved for epoch ", epoch)

class lstmwrapper(nn.Module):
    def __init__(self,input_size=47764, output_size=3620,hidden_size=128,num_layers=16,batch_first=True,
                 dropout=True):
        super(lstmwrapper, self).__init__()
        self.lstm=LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,
                       batch_first=batch_first,dropout=dropout)
        self.output=nn.Linear(hidden_size,output_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.output.reset_parameters()

    def forward(self, input, hx=None):
        output,statetuple=self.lstm(input,hx)
        return self.output(output)


def main():
    with torch.cuda.device(1):
        story_limit = 150
        epoch_batches_count = 64
        epochs_count = 1024
        lr = 1e-1
        optim = None
        starting_epoch = -1
        bs=32
        pgd = PreGenData(bs)

        task_dir = os.path.dirname(abspath(__file__))
        processed_data_dir = join(task_dir, 'data',"processed")
        lexicon_dictionary=pickle.load( open(join(processed_data_dir, 'lexicon-dict.pkl'), 'rb'))
        x=len(lexicon_dictionary)

        computer = lstmwrapper(input_size=x,output_size=x,hidden_size=256,num_layers=128,batch_first=True,
                               dropout=True)
        computer.reset_parameters()

        # if load model
        # computer, optim, starting_epoch = load_model(computer)

        computer = computer.cuda()
        # if optim is None:
        #     optimizer = torch.optim.Adam(computer.parameters(), lr=lr)
        # else:
        print('use Adadelta optimizer with learning rate ', lr)
        optimizer = torch.optim.SGD(computer.parameters(), lr=lr)

        # starting with the epoch after the loaded one
        train(computer, optimizer, story_limit, bs, pgd, x, int(starting_epoch) + 1, epochs_count, epoch_batches_count)


if __name__ == "__main__":
    main()
