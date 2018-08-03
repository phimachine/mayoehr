import pandas as pd
import torch
import numpy as np
import pdb
from pathlib import Path
import os
from os.path import abspath
from death.post.inputgen_planD import InputGenD, train_valid_split
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.modules import LSTM
from torch.autograd import Variable
import pickle
from shutil import copy
import traceback
from collections import deque
import datetime

batch_size = 1

class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

def save_model(net, optim, epoch, iteration):
    epoch = int(epoch)
    task_dir = os.path.dirname(abspath(__file__))
    pickle_file = Path(task_dir).joinpath("lstmsaves/lstm_" + str(epoch) +  "_" + str(iteration) + ".pkl")
    pickle_file = pickle_file.open('wb')
    torch.save((net,  optim, epoch, iteration), pickle_file)
    print('model saved at', pickle_file)

def load_model(computer, optim, starting_epoch, starting_iteration):
    task_dir = os.path.dirname(abspath(__file__))
    save_dir = Path(task_dir) / "lstmsaves"
    highestepoch = 0
    highestiter = 0
    for child in save_dir.iterdir():
        epoch = str(child).split("_")[3]
        iteration = str(child).split("_")[4].split('.')[0]
        iteration = int(iteration)
        epoch = int(epoch)
        # some files are open but not written to yet.
        if child.stat().st_size > 20480:
            if epoch > highestepoch or (iteration > highestiter and epoch==highestepoch):
                highestepoch = epoch
                highestiter = iteration
    if highestepoch == 0 and highestiter == 0:
        print("nothing to load")
        return computer, optim, starting_epoch, starting_iteration
    pickle_file = Path(task_dir).joinpath("lstmsaves/lstm_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
    print("loading model at", pickle_file)
    pickle_file = pickle_file.open('rb')
    computer, optim, epoch, iteration = torch.load(pickle_file)
    print('Loaded model at epoch ', highestepoch, 'iteartion', iteration)
    #
    # for child in save_dir.iterdir():
    #     epoch = str(child).split("_")[3].split('.')[0]
    #     iteration = str(child).split("_")[4].split('.')[0]
    #     if int(epoch) != highestepoch and int(iteration) != highestiter:
    #         # TODO might have a bug here.
    #         os.remove(child)
    # print('Removed incomplete save file and all else.')

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
          train, valid_iterator, starting_epoch, total_epochs, starting_iter, iter_per_epoch, logfile=False):
    print_interval=10
    val_interval=50
    save_interval=400
    target_dim=None
    rldmax_len=50
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
                running_loss_deque.appendleft(printloss)
                if i % print_interval == 0:
                    running_loss=np.mean(running_loss_deque)
                    if logfile:
                        with open(logfile, 'a') as handle:
                            handle.write("learning.   count: %4d, training loss: %.10f \n" %
                                         (i, printloss))
                    print("learning.   count: %4d, training loss: %.10f" %
                          (i, printloss))
                    if i!=0:
                        print("count: %4d, running loss: %.10f" % (i, running_loss))

                if i % val_interval == 0:
                    # we should consider running validation multiple times and average. TODO
                    (input,target,loss_type)=next(valid_iterator)
                    val_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                                   real_criterion, binary_criterion, validate=True)
                    if val_loss is not None:
                        printloss = float(val_loss[0])
                    if logfile:
                        with open(logfile, 'a') as handle:
                            handle.write("validation. count: %4d, val loss     : %.10f \n" %
                                         (i, printloss))
                    print("validation. count: %4d, training loss: %.10f" %
                          (i, printloss))

                if i % save_interval == 0:
                    save_model(computer, optimizer, epoch, i)
                    print("model saved for epoch", epoch, "input", i)
            else:
                break

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

def main(load=False):
    total_epochs = 10
    iter_per_epoch = 100000
    lr = 1e-2
    optim = None
    starting_epoch = 0
    starting_iteration= 0
    logfile = "log.txt"

    num_workers = 32
    ig = InputGenD()
    # multiprocessing disabled, because socket request seems unstable.
    # performance should not be too bad?
    trainds,validds=train_valid_split(ig,split_fold=10)
    traindl = DataLoader(dataset=trainds, batch_size=1, num_workers=num_workers)
    validdl = DataLoader(dataset=validds, batch_size=1)
    print("Using", num_workers, "workers for training set")
    # testing whether this LSTM works is basically a question whether
    lstm=lstmwrapper(input_size=47764,hidden_size=512,num_layers=128,batch_first=True,
                     dropout=True)

    # load model:
    if load:
        print("loading model")
        lstm, optim, starting_epoch, starting_iteration = load_model(lstm, optim, starting_epoch, starting_iteration)

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
          traindl, iter(validdl), int(starting_epoch), total_epochs,int(starting_iteration), iter_per_epoch, logfile)


if __name__ == "__main__":
    main()
