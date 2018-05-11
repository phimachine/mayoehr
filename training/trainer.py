from training.datagen import PreGenData
from archi.computer import Computer
import torch
import numpy
import archi.param as param
import pdb
from pathlib import Path
import os
from os.path import abspath
import time
# task 10 of babi

batch_size=param.bs

class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

def save_model(net, optim, epoch):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    task_dir = os.path.dirname(abspath(__file__))
    print(task_dir)
    pickle_file=Path("saves/DNC_"+str(epoch)+".pkl")
    pickle_file=pickle_file.open('w')

    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        pickle_file)

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
        computer.new_sequence_reset()

    return story_loss

def train(computer, optimizer, story_length, batch_size, pgd):
    train_loss_history=[]
    test_history=[]
    for epoch in range(epochs_count):

        running_loss=0

        for batch in range(epoch_batches_count):

            train_story_loss=run_one_story(computer, optimizer, story_length, batch_size, pgd)
            print("learning. epoch: %4d, batch number: %4d, training loss: %.4f" %
                  (epoch+1, batch+1, train_story_loss.item()))
            running_loss+=train_story_loss
            val_freq=64
            if batch%val_freq==val_freq-1:
                print('summary.  epoch: %4d, batch number: %4d, running loss: %.4f' %
                      (epoch + 1, batch + 1, running_loss / val_freq))
                running_loss=0
                # also test the model
                val_loss=run_one_story(computer, optimizer, story_length, batch_size, pgd, validate=False)
                print('validate. epoch: %4d, batch number: %4d, validation loss: %.4f' %
                      (epoch + 1, batch + 1, val_loss))
                test_history+=[val_loss]

            train_loss_history+=[train_story_loss]

        save_model(computer,optimizer,epoch)
        print("model saved for epoch ", epoch)



if __name__=="__main__":

    story_limit=150
    epoch_batches_count=32
    epochs_count=1024
    lr=1e-5
    pgd=PreGenData(param.bs)
    computer=Computer()
    computer=computer.cuda()
    optimizer=torch.optim.Adam(computer.parameters(),lr=lr)

    train(computer,optimizer,story_limit, batch_size, pgd)