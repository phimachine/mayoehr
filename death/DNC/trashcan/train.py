from death.DNC.archi.computer import Computer
import torch
import numpy
import pdb
from pathlib import Path
import os
from os.path import abspath
import death.DNC.archi.param as param

batch_size=param.bs

class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

def save_model(net,optim,epoch):
    epoch=int(epoch)
    task_dir = os.path.dirname(abspath(__file__))
    pickle_file=Path(task_dir).joinpath("saves/DNCfull_"+str(epoch)+".pkl")
    pickle_file=pickle_file.open('wb')
    torch.save((net,optim,epoch),pickle_file)

def load_model(computer):
    task_dir = os.path.dirname(abspath(__file__))
    save_dir=Path(task_dir)/"saves"
    highestepoch=-1
    for child in save_dir.iterdir():
        epoch=str(child).split("_")[2].split('.')[0]
        epoch=int(epoch)
        # some files are open but not written to yet.
        if epoch > highestepoch and child.stat().st_size>2048:
            highestepoch=epoch
    if highestepoch==-1:
        return computer, None, -1
    pickle_file=Path(task_dir).joinpath("saves/DNCfull_"+str(highestepoch)+".pkl")
    print("loading model at ", pickle_file)
    pickle_file=pickle_file.open('rb')
    model, optim, epoch=torch.load(pickle_file)

    print('Loaded model at epoch ', highestepoch)

    for child in save_dir.iterdir():
        epoch=str(child).split("_")[2].split('.')[0]
        if int(epoch)!=highestepoch:
            os.remove(child)
    print('Removed incomplete save file and all else.')

    return model, optim, epoch


def run_one_story(computer, optimizer, story_length, batch_size, pgd, validate=False):
    # to promote code reuse
    if not validate:
        input_data, target_output, critical_index=pgd.get_train()
    else:
        input_data, target_output, critical_index=pgd.get_validate()

    input_data=torch.Tensor(input_data).cuda()
    target_output=torch.Tensor(target_output).cuda()
    stairs=torch.Tensor(numpy.arange(0,param.bs*story_length,story_length))
    critical_index=critical_index+stairs.unsqueeze(1)
    critical_index=critical_index.view(-1)
    critical_index=critical_index.long().cuda()

    criterion=torch.nn.CrossEntropyLoss()

    with torch.no_grad if validate else dummy_context_mgr():

        story_output = torch.Tensor(batch_size, story_length, param.x).cuda()
        computer.new_sequence_reset()
        # a single story
        for timestep in range(story_length):
            # feed the batch into the machine
            # Expected input dimension: (150, 27)
            # output: (150,27)
            batch_input_of_same_timestep = input_data[:, timestep, :]

            # usually batch does not interfere with each other's logic
            batch_output=computer(batch_input_of_same_timestep)
            if torch.isnan(batch_output).any():
                pdb.set_trace()
                raise ValueError("nan is found in the batch output.")
            story_output[:, timestep,:] = batch_output

        target_output=target_output.view(-1)
        story_output=story_output.view(-1,param.x)
        story_output=story_output[critical_index,:]
        target_output=target_output[critical_index].long()

        story_loss = criterion(story_output, target_output)
        if not validate:
            # I chose to backward a derivative only after a whole story has been taken in
            # This should lead to a more stable, but initially slower convergence.
            story_loss.backward()
            optimizer.step()

    return story_loss


def train(computer, optimizer, story_length, batch_size, pgd, starting_epoch):
    for epoch in range(starting_epoch, epochs_count):

        running_loss = 0

        for batch in range(epoch_batches_count):

            train_story_loss = run_one_story(computer, optimizer, story_length, batch_size, pgd)
            print("learning. epoch: %4d, batch number: %4d, training loss: %.4f" %
                  (epoch, batch, train_story_loss.item()))
            running_loss += train_story_loss
            val_freq = 16
            if batch % val_freq == val_freq - 1:
                print('summary.  epoch: %4d, batch number: %4d, running loss: %.4f' %
                      (epoch, batch, running_loss / val_freq))
                running_loss = 0
                # also test the model
                val_loss = run_one_story(computer, optimizer, story_length, batch_size, pgd, validate=False)
                print('validate. epoch: %4d, batch number: %4d, validation loss: %.4f' %
                      (epoch, batch, val_loss))

        save_model(computer, optimizer, epoch)
        print("model saved for epoch ", epoch)


if __name__ == "__main__":

    story_limit = 150
    epoch_batches_count = 64
    epochs_count = 1024
    lr = 1e-5
    pgd = PreGenData(param.bs)
    computer = Computer()
    optim = None
    starting_epoch = -1

    # if load model
    computer, optim, starting_epoch = load_model(computer)

    computer = computer.cuda()
    if optim is None:
        optimizer = torch.optim.Adam(computer.parameters(), lr=lr)
    else:
        print('use Adadelta optimizer with learning rate ', lr)
        optimizer = torch.optim.Adadelta(computer.parameters(), lr=lr)

    # starting with the epoch after the loaded one
    train(computer, optimizer, story_limit, batch_size, pgd, int(starting_epoch) + 1)
