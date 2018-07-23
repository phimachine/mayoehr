
from os.path import abspath
import os
from pathlib import Path
import shutil
import torch
import pickle

def save_model(net, optim, epoch, iteration):
    epoch = int(epoch)
    task_dir = os.path.dirname(abspath(__file__))
    data_dir = Path(task_dir) / "saves"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    pickle_path = Path(task_dir).joinpath("saves/DNCfull_" + str(epoch) + "_" + str(iteration) + ".pkl")
    pickle_file = pickle_path.open('wb')
    torch.save((net, optim, epoch, iteration), pickle_file)
    print('model saved at', pickle_path)


def save_model_old(net, optim, epoch, iteration):
    print("saving model")
    epoch = int(epoch)
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    task_dir = os.path.dirname(abspath(__file__))
    pickle_file = Path(task_dir).joinpath("saves/DNCfull_" + str(epoch) + "_" + str(iteration) + ".pkl")
    fhand = pickle_file.open('wb')
    try:
        pickle.dump((state_dict, optim, epoch, iteration), fhand)
        print('model saved at', pickle_file)
    except:
        fhand.close()
        os.remove(pickle_file)


def load_model(computer, remove=True):
    task_dir = os.path.dirname(abspath(__file__))
    save_dir = Path(task_dir) / "saves"
    highestepoch = 0
    highestiter = 0
    for child in save_dir.iterdir():
        epoch = str(child).split("_")[3]
        iteration = str(child).split("_")[4].split('.')[0]
        iteration = int(iteration)
        epoch = int(epoch)
        # some files are open but not written to yet.
        if epoch > highestepoch and iteration > highestiter and child.stat().st_size > 20480:
            highestepoch = epoch
            highestiter = iteration
    if highestepoch == 0 and highestiter == 0:
        print("Nothing to load.")
        return computer, None, 0, 0
    pickle_file = Path(task_dir).joinpath("saves/DNCfull_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
    print("loading model at ", pickle_file)
    pickle_file = pickle_file.open('rb')
    computer, optim, epoch, iteration = torch.load(pickle_file)
    print('Loaded model at epoch ', highestepoch, 'iteartion', iteration)

    if remove:
        for child in save_dir.iterdir():
            epoch = str(child).split("_")[3].split('.')[0]
            iteration = str(child).split("_")[4].split('.')[0]
            if int(epoch) != highestepoch and int(iteration) != highestiter:
                # TODO might have a bug here.
                os.remove(child)
                print("removed saved weighting", child)
        print('Removed incomplete save file and all else.')

    return computer, optim, highestepoch, highestiter


def load_model_old(computer):
    task_dir = os.path.dirname(abspath(__file__))
    save_dir = Path(task_dir) / "saves"
    highestepoch = 0
    highestiter = 0
    for child in save_dir.iterdir():
        epoch = str(child).split("_")[3]
        iteration = str(child).split("_")[4].split('.')[0]
        iteration = int(iteration)
        epoch = int(epoch)
        # some files are open but not written to yet.
        if epoch > highestepoch and iteration > highestiter and child.stat().st_size > 204800:
            highestepoch = epoch
            highestiter = iteration
    if highestepoch == 0 and highestepoch == 0:
        return computer, None, 0, 0
    pickle_file = Path(task_dir).joinpath("saves/DNCfull_" + str(highestepoch) + "_" + str(iteration) + ".pkl")
    print("loading model at ", pickle_file)
    pickle_file = pickle_file.open('rb')
    modelsd, optim, epoch, iteration = torch.load(pickle_file)
    computer.load_state_dict(modelsd)
    print('Loaded model at epoch ', highestepoch, 'iteartion', iteration)

    for child in save_dir.iterdir():
        epoch = str(child).split("_")[3].split('.')[0]
        iteration = str(child).split("_")[4].split('.')[0]
        if int(epoch) != highestepoch and int(iteration) != highestiter:
            os.remove(child)
    print('Removed incomplete save file and all else.')

    return computer, optim, epoch, iteration


def salvage():
    # this function will pick up the last two highest epoch training and save them somewhere else,
    # this is to prevent unexpected data loss.
    # We are working in a /tmp folder, and we write around 1Gb per minute.
    # The loss of data is likely.

    task_dir = os.path.dirname(abspath(__file__))
    save_dir = Path(task_dir) / "saves"
    highestepoch = 0
    secondhighestiter = 0
    highestiter = 0
    for child in save_dir.iterdir():
        epoch = str(child).split("_")[3]
        iteration = str(child).split("_")[4].split('.')[0]
        iteration = int(iteration)
        epoch = int(epoch)
        # some files are open but not written to yet.
        if epoch > highestepoch and iteration > highestiter and child.stat().st_size > 20480:
            highestepoch = epoch
            highestiter = iteration
    if highestepoch == 0 and highestiter == 0:
        print("no file to salvage")
        return
    if secondhighestiter != 0:
        pickle_file2 = Path(task_dir).joinpath(
            "saves/DNCfull_" + str(highestepoch) + "_" + str(secondhighestiter) + ".pkl")
        shutil.copy(pickle_file2, "/infodev1/rep/projects/jason/pickle/salvage2.pkl")

    pickle_file1 = Path(task_dir).joinpath("saves/DNCfull_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
    shutil.copy(pickle_file1, "/infodev1/rep/projects/jason/pickle/salvage1.pkl")

    print('salvaged, we can start again with /infodev1/rep/projects/jason/pickle/salvage1.pkl')


def log_print(string, logfile):
    if logfile:
        with open(logfile, 'a') as handle:
            handle.write(string + "\n")
    print(string)


def get_log_file():
    # returns a path to a non-overlapping log file
    log_dir=Path("log")
    highest_log = 0

    for child in log_dir.iterdir():
        log_num=str(child).split("_")[1].split('.')[0]
        if log_num> highest_log:
            highest_log=log_num

    highest_log+=1
    return log_dir/("log"+str(highest_log)+".txt")