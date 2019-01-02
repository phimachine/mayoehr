# I decide to run my model on babi again to see if the convergen ce problem is with my model or dataset

from dnc import DNC
import torch
import numpy
from pathlib import Path
import pickle
import os
from os.path import join, abspath
from death.DNC.trashcan.babi.babigen import PreGenData
from torch.autograd import Variable


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


def run_one_story(computer, optimizer, story_length, batch_size, pgd, input_dim, mhx, validate=False):
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

        # a single story
        story_output, (_, mhx, _) = computer(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True)
        if (story_output!=story_output).any():
            raise ValueError("nan is found in the batch output.")

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

    return story_loss, mhx


def train(computer, optimizer, story_length, batch_size, pgd, input_dim, starting_epoch, epochs_count, epoch_batches_count):
    mhx=None

    for epoch in range(starting_epoch, epochs_count):

        for batch in range(epoch_batches_count):

            train_story_loss, mhx = run_one_story(computer, optimizer, story_length, batch_size, pgd, input_dim,mhx)
            print("learning. epoch: %4d, batch number: %4d, training loss: %.4f" %
                  (epoch, batch, train_story_loss[0]))
            # keeping the running loss causes GPU memory leak.
            # reassignment of variables retain graph
            # reassignment with 0 changes the internal value and does not seem to reinitiate the object?
            # do not keep running loss. Not necessary anyway.
            val_freq = 16
            if batch % val_freq == val_freq - 1:
                # also test the model
                val_loss = run_one_story(computer, optimizer, story_length, batch_size, pgd, input_dim, mhx, validate=False)
                print('validate. epoch: %4d, batch number: %4d, validation loss: %.4f' %
                      (epoch, batch, val_loss))

        save_model(computer, optimizer, epoch)
        print("model saved for epoch ", epoch)


def main():
    story_limit = 150
    epoch_batches_count = 64
    epochs_count = 1024
    lr = 1e-11
    optim = 1
    starting_epoch = -1
    bs=32
    pgd = PreGenData(bs)

    task_dir = os.path.dirname(abspath(__file__))
    processed_data_dir = join(task_dir, 'data',"processed")
    lexicon_dictionary=pickle.load( open(join(processed_data_dir, 'lexicon-dict.pkl'), 'rb'))
    x=len(lexicon_dictionary)


    computer = DNC(x,x,num_layers=4,num_hidden_layers=4,cell_size=4,nr_cells=4,read_heads=4,gpu_id=0).cuda()

    # if load model
    # computer, optim, starting_epoch = load_model(computer)

    computer = computer.cuda()
    if optim is None:
        optimizer = torch.optim.Adam(computer.parameters(), lr=lr)
    else:
        print('use Adadelta optimizer with learning rate ', lr)
        optimizer = torch.optim.Adadelta(computer.parameters(), lr=lr)

    # starting with the epoch after the loaded one
    train(computer, optimizer, story_limit, bs, pgd, x, int(starting_epoch) + 1, epochs_count, epoch_batches_count)


if __name__ == "__main__":
    main()
