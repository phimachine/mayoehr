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
from death.DNC.tsDNCtrainer import logprint
from death.taco.collate import pad_collate

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
    pickle_file = Path(task_dir).joinpath("saves/" + savestr + "/lstmnorm_" + str(epoch) + "_" + str(iteration) + ".pkl")
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
        "saves/" + savestr + "/lstmnorm_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
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

global_exception_counter=0
def run_one_patient(computer, input, target, target_dim, optimizer, loss_type, real_criterion,
                    binary_criterion, validate=False):
    global global_exception_counter
    patient_loss=None
    try:
        optimizer.zero_grad()
        input = Variable(torch.Tensor(input).cuda())
        target = Variable(torch.Tensor(target).cuda())

        # we have no critical index, becuase critical index are those timesteps that
        # DNC is required to produce outputs. This is not the case for our project.
        # criterion does not need to be reinitiated for every story, because we are not using a mask

        patient_output=computer(input)
        cause_of_death_output = patient_output[:, :, 1:]
        cause_of_death_target = target[:, :, 1:]
        patient_loss= binary_criterion(cause_of_death_output, cause_of_death_target)

        if not validate:
            patient_loss.backward()
            optimizer.step()

        if global_exception_counter>-1:
            global_exception_counter-=1
    except ValueError:
        traceback.print_exc()
        print("Value Error reached")
        print(datetime.datetime.now().time())
        global_exception_counter+=1
        if global_exception_counter==10:
            save_model(computer,optimizer,epoch=0,iteration=global_exception_counter)
            raise ValueError("Global exception counter reached 10. Likely the model has nan in weights")
        else:
            pass

    return patient_loss


def train(computer, optimizer, real_criterion, binary_criterion,
          train, valid_dl, starting_epoch, total_epochs, starting_iter, iter_per_epoch, savestr, logfile=False):
    valid_iterator=iter(valid_dl)
    print_interval=10
    val_interval=200
    save_interval=800
    target_dim=None
    rldmax_len=50
    val_batch=100
    running_loss_deque=deque(maxlen=rldmax_len)
    if logfile:
        open(logfile, 'w').close()

    for epoch in range(starting_epoch, total_epochs):
        for i, (input, target, loss_type) in enumerate(train):
            i=starting_iter+i
            if target_dim is None:
                target_dim=target.shape[2]

            if i < iter_per_epoch:
                train_story_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                                   real_criterion, binary_criterion)
                if train_story_loss is not None:
                    printloss=float(train_story_loss[0])
                else:
                    raise ValueError("Why would story loss be None?")
                running_loss_deque.appendleft(printloss)
                if i % print_interval == 0:
                    running_loss=np.mean(running_loss_deque)
                    logprint(logfile, "learning.   count: %4d, training loss: %.10f, running loss: %.10f" %
                             (i, printloss, running_loss))


                if i % val_interval == 0:
                    printloss=0
                    for _ in range(val_batch):
                        # we should consider running validation multiple times and average. TODO
                        try:
                            (input,target,loss_type)=next(valid_iterator)
                        except StopIteration:
                            valid_iterator=iter(valid_dl)
                            (input,target,loss_type)=next(valid_iterator)

                        val_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                                       real_criterion, binary_criterion, validate=True)
                        if val_loss is not None:
                            printloss += float(val_loss[0])
                        else:
                            raise ValueError ("Investigate this")
                    printloss=printloss/val_batch
                    logprint(logfile, "validation. count: %4d, val loss     : %.10f" %
                             (i, printloss))

                if i % save_interval == 0:
                    save_model(computer, optimizer, epoch, i, savestr)
                    print("model saved for epoch", epoch, "input", i)
            else:
                break

def valid(computer, optimizer, real_criterion, binary_criterion,
          train, valid_dl, starting_epoch, total_epochs, starting_iter, iter_per_epoch, logfile=False):
    running_loss=[]
    target_dim=None
    valid_iterator=iter(valid_dl)

    for i in valid_iterator:
        input, target, loss_type=next(valid_iterator)
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


class lstmwrapper(nn.Module):
    def __init__(self,input_size=66529, output_size=5952,hidden_size=52,num_layers=16,batch_first=True,
                 dropout=True):
        super(lstmwrapper, self).__init__()
        self.lstm=LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,
                       batch_first=batch_first,dropout=dropout)
        self.output=nn.Linear(hidden_size,output_size)
        self.bn = nn.BatchNorm1d(input_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.output.reset_parameters()

    def forward(self, input, hx=None):
        input=self.bn(input)
        output,statetuple=self.lstm(input,hx)
        return self.output(output)

class lstmwrapper2(nn.Module):
    def __init__(self,input_size=66529, output_size=5952,hidden_size=52,num_layers=16,batch_first=True,
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

def validationonly():
    '''
    :return:
    '''

    lr = 1e-2
    optim = None
    logfile = "vallog.txt"

    num_workers = 8
    ig = InputGenH()
    # multiprocessing disabled, because socket request seems unstable.
    # performance should not be too bad?
    trainds = ig.get_train()
    validds = ig.get_valid()
    validdl = DataLoader(dataset=validds,num_workers=num_workers, batch_size=1)
    print("Using", num_workers, "workers for validation set")
    # testing whether this LSTM works is basically a question whether
    lstm = lstmwrapper()

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

def main(load,savestr):
    total_epochs = 3
    iter_per_epoch = 100000
    lr = 1e-3
    optim = None
    starting_epoch = 0
    starting_iteration= 0
    logstring = str(datetime.datetime.now().time())
    logstring.replace(" ", "_")
    logfile = "log/"+savestr+"_"+logstring+".txt"

    num_workers = 8
    ig = InputGenF(death_fold=0)
    trainds = ig.get_train()
    validds = ig.get_valid()
    validdl = DataLoader(dataset=validds, batch_size=8, num_workers=num_workers, collate_fn=pad_collate)
    traindl = DataLoader(dataset=trainds, batch_size=8, num_workers=num_workers//4, collate_fn=pad_collate)

    print("Using", num_workers, "workers for training set")
    # testing whether this LSTM works is basically a question whether
    lstm=lstmwrapper()

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
    main()

    '''
    lr=1e-4 is extremely slow.
    This is probably because I averaged the loss across the whole sequence? Why is this not a problem for Tacotron?
    '''