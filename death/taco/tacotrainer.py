"""
Tacotrainer is modified from trainerD3.
This will involve another batch solution I think. I will need to define a collate function.
"""
import pandas
import torch
import numpy as np
from pathlib import Path
import os
from os.path import abspath
from death.post.inputgen_planG import InputGenG, pad_collate
from torch.utils.data import DataLoader
import torch.nn as nn
from death.taco.model import Tacotron
from torch.autograd import Variable
import pickle
from shutil import copy
import traceback
from collections import deque
import datetime
from death.DNC.seqtrainer import logprint, datetime_filename

global_exception_counter = 0
i = None
debug=True

def sv(var):
    return var.data.cpu().numpy()

class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def save_model(net, optim, epoch, iteration, savestr):
    if epoch!=0:
        print("what is this?")
    epoch = int(epoch)
    task_dir = os.path.dirname(abspath(__file__))
    if not os.path.isdir(Path(task_dir)/"saves"/savestr):
        os.mkdir(Path(task_dir)/"saves"/savestr)
    pickle_file = Path(task_dir).joinpath("saves/" + savestr + "/taco_" + str(epoch) + "_" + str(iteration) + ".pkl")
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
        "saves/" + savestr + "/taco_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
    print("loading model at",pickle_file)
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
    save_dir = Path(task_dir) / "saves"/savestr
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
            "saves/DNC"+savestr+"_" + str(highestepoch) + "_" + str(secondhighestiter) + ".pkl")
        copy(pickle_file2, "/infodev1/rep/projects/jason/pickle/salvage2.pkl")

    pickle_file1 = Path(task_dir).joinpath("saves/DNC"+savestr+"_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
    copy(pickle_file1, "/infodev1/rep/projects/jason/pickle/salvage1.pkl")

    print('salvaged, we can start again with /infodev1/rep/projects/jason/pickle/salvage1.pkl')


def run_one_patient(computer, input, target, target_dim, optimizer, loss_type, real_criterion,
                    binary_criterion, validate=False):
    global global_exception_counter
    patient_loss=None
    try:
        if not validate:
            computer.train()
        else:
            computer.eval()
        optimizer.zero_grad()
        input = Variable(torch.Tensor(input).cuda())
        target = Variable(torch.Tensor(target).cuda())

        # we have no critical index, becuase critical index are those timesteps that
        # criterion does not need to be reinitiated for every story, because we are not using a mask

        patient_output=computer(input)
        cause_of_death_output = patient_output[:, 1:]
        cause_of_death_target = target[:, 1:]
        # pdb.set_trace()
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
          train, valid, starting_epoch, total_epochs, starting_iter, iter_per_epoch, savestr, logfile=False):
    global global_exception_counter
    valid_iterator=iter(valid)
    print_interval = 10
    val_interval = 400
    save_interval = 800
    target_dim = None
    rldmax_len = 50
    val_batch=100
    running_loss_deque = deque(maxlen=rldmax_len)
    if logfile:
        open(logfile, 'w+').close()
    global i

    for epoch in range(starting_epoch, total_epochs):
        for i, (input, target, loss_type) in enumerate(train):
            i = starting_iter + i
            if target_dim is None:
                target_dim = target.shape[1]

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
                    logprint(logfile, "learning.   count: %4d, training loss: %.10f, running loss: %.10f" %
                             (i, printloss, running_loss))

                if i % val_interval == 0:
                    printloss = 0
                    for _ in range(val_batch):
                        try:
                            (input, target, loss_type) = next(valid_iterator)
                        except StopIteration:
                            valid_iterator=iter(valid)
                            (input, target, loss_type) = next(valid_iterator)
                        val_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                                   real_criterion, binary_criterion, validate=True)
                        if val_loss is not None:
                            printloss += float(val_loss[0])
                    printloss=printloss/val_batch
                    logprint(logfile, "validation. count: %4d, val loss     : %.10f" %
                             (i, printloss))

                if i % save_interval == 0:
                    save_model(computer, optimizer, epoch, i, savestr)
                    print("model saved for epoch", epoch, "input", i)
            else:
                break
        starting_iter=0

def valid_only(load=True, lr=1e-3, savestr=""):
    lr = lr
    optim = None
    starting_epoch = 0
    starting_iteration = 0
    logfile = "log/"+savestr+"_"+datetime_filename()+".txt"
    target_dim=None

    num_workers = 16
    ig = InputGenG()
    validds = ig.get_valid()
    valid = DataLoader(dataset=validds, batch_size=8, num_workers=num_workers, collate_fn=pad_collate)
    valid_iterator=iter(valid)

    print("Using", num_workers, "workers for training set")
    computer = Tacotron()

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

    val_batch=500
    printloss = 0
    for _ in range(val_batch):
        try:
            (input, target, loss_type) = next(valid_iterator)
        except StopIteration:
            valid_iterator = iter(valid)
            (input, target, loss_type) = next(valid_iterator)
        if target_dim is None:
            target_dim = target.shape[1]
        val_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                   real_criterion, binary_criterion, validate=True)
        if val_loss is not None:
            printloss += float(val_loss[0])
    printloss = printloss / val_batch
    logprint(logfile, "validation. count: %4d, val loss     : %.10f" %
             (val_batch, printloss))

def forevermain(load=False, lr=1e-3, savestr=""):
    print("Will run main() forever in a loop.")
    while True:
        try:
            main(load, lr, savestr)
        except ValueError:
            traceback.print_exc()

def main(load=False, lr=1e-3, savestr=""):
    total_epochs = 10
    iter_per_epoch = 10000
    lr = lr
    optim = None
    starting_epoch = 0
    starting_iteration = 0
    logfile = "log/"+savestr+"_"+datetime_filename()+".txt"

    num_workers = 16
    ig = InputGenG()
    validds = ig.get_valid()
    trainds = ig.get_train()
    validdl = DataLoader(dataset=validds, batch_size=8, num_workers=num_workers//4, collate_fn=pad_collate)
    traindl = DataLoader(dataset=trainds, batch_size=8, num_workers=num_workers, collate_fn=pad_collate)

    print("Using", num_workers, "workers for training set")
    computer = Tacotron()

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
        for group in optimizer.param_groups:
            print("Currently using a learing rate of ", group["lr"])

    real_criterion = nn.SmoothL1Loss()
    # time-wise sum, label-wise average.
    binary_criterion = nn.BCEWithLogitsLoss()

    # starting with the epoch after the loaded one

    train(computer, optimizer, real_criterion, binary_criterion,
          traindl, validdl, int(starting_epoch), total_epochs, int(starting_iteration), iter_per_epoch, savestr,
          logfile)


if __name__ == "__main__":
    main(load=True, savestr="taco")

"""
Training was run for 10 hours, for 10 epochs.
The performance however, is not so good. In the end the moving average loss is around the same with what it achieved
within 10 minutes. What does it mean? Wrong model again? Does not seem like any information is extracted.
If you give it literally anything with a backprop, this is what you're gonna get.
"""

'''
0.0004722830 is the validation with 500 batch
0.00055 is LSTM
0.00030 is DNC
DNC > Tacotron > LSTM
'''