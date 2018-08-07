import torch
from pathlib import Path
import os
from os.path import abspath
from death.post.inputgen_planC import InputGen
from torch.utils.data import DataLoader
import torch.nn as nn
from death.DNC.trashcan.frankenstein import Frankenstein as DNC
from torch.autograd import Variable
import gc
import pickle

batch_size = 1


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

# def save_model2(net, optim, epoch, iteration):
#
#     print("saving model")
#     epoch = int(epoch)
#     task_dir = os.path.dirname(abspath(__file__))
#     pickle_file = Path(task_dir).joinpath("saves/DNCfull_" + str(epoch)+ "_"+str(iteration) + ".pkl")
#     fhand = pickle_file.open('wb')
#     try:
#         pickle.dump((net,optim, epoch, iteration),fhand)
#         print('model saved')
#     except:
#         fhand.close()
#         os.remove(pickle_file)

def save_model(net, optim, epoch, iteration):
    print("saving model")
    epoch = int(epoch)
    task_dir = os.path.dirname(abspath(__file__))
    pickle_file = Path(task_dir).joinpath("saves/DNCfull_" + str(epoch) + "_" + str(iteration) + ".pkl")
    fhand = pickle_file.open('wb')
    try:
        pickle.dump((net.state_dict(),optim, epoch, iteration),fhand)
        print('model saved')
    except:
        fhand.close()
        os.remove(pickle_file)

def load_model(computer):
    task_dir = os.path.dirname(abspath(__file__))
    save_dir = Path(task_dir) / "saves"
    highestepoch = -1
    highestiter = -1
    for child in save_dir.iterdir():
        epoch = str(child).split("_")[3]
        iteration = str(child).split("_")[4].split('.')[0]
        iteration=int(iteration)
        epoch = int(epoch)
        # some files are open but not written to yet.
        if epoch > highestepoch and iteration>highestiter and child.stat().st_size > 204800:
            highestepoch = epoch
            highestiter=iteration
    if highestepoch == -1 and highestepoch==-1:
        return computer, None, -1, -1
    pickle_file = Path(task_dir).joinpath("saves/DNCfull_" + str(highestepoch)+"_"+str(iteration) + ".pkl")
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

def run_one_patient_one_step():
    # this is so python does garbage collection automatically.
    # we are debugging the
    pass

def run_one_patient(computer, input, target, target_dim, optimizer, loss_type, real_criterion,
                    binary_criterion, validate=False):

    input = Variable(torch.Tensor(input).cuda())
    target = Variable(torch.Tensor(target).cuda())

    # we have no critical index, becuase critical index are those timesteps that
    # DNC is required to produce outputs. This is not the case for our project.
    # criterion does not need to be reinitiated for every story, because we are not using a mask

    time_length = input.size()[1]
    # with torch.no_grad if validate else dummy_context_mgr():
    patient_output = Variable(torch.Tensor(1, time_length, target_dim)).cuda()
    for timestep in range(time_length):
        # first colon is always size 1
        feeding = input[:, timestep, :]
        output = computer(feeding)
        assert not (output!=output).any()
        patient_output[0, timestep, :] = output

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
        underestimation = nn.functional.relu(underestimation)
        toe_loss = real_criterion(underestimation,torch.zeros_like(underestimation).cuda())
        cod_loss = binary_criterion(cause_of_death_output,cause_of_death_target)
        patient_loss=toe_loss+cod_loss

    if not validate:
        # TODO UNDERSTAND WHAT THE FLAG MEANS
        patient_loss.backward()
        optimizer.step()

    del input
    del target

    return patient_loss


def train(computer, optimizer, real_criterion, binary_criterion,
          igdl, starting_epoch, total_epochs, iter_per_epoch, target_dim, logfile=False):
    if logfile:
        open(logfile,'w').close()

    for epoch in range(starting_epoch, total_epochs):

        for i, (input, target, loss_type) in enumerate(igdl):

            if i < iter_per_epoch:
                train_story_loss = run_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                                   real_criterion,binary_criterion)
                computer.new_sequence_reset()
                gc.collect()
                del input, target, loss_type
                if i % 1 == 0:
                    if logfile:
                        with open(logfile,'a') as handle:
                            handle.write("learning. count: %4d, training loss: %.4f \n" %
                                          (i, train_story_loss[0]))
                        print("learning. count: %4d, training loss: %.4f" %
                              (i, train_story_loss[0]))
                    else:
                        print("learning. count: %4d, training loss: %.4f" %
                              (i, train_story_loss[0]))


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
                if i % 1000 == 999:
                    save_model(computer, optimizer, epoch, i)
                    print("model saved for epoch", epoch, "input", i)
            else:
                break


def main():
    total_epochs = 10
    iter_per_epoch=100000
    lr = 1e-20
    optim = None
    starting_epoch = -1
    target_dim=3656
    logfile="log.txt"

    num_workers=8
    ig=InputGen()
    igdl=DataLoader(dataset=ig,batch_size=1,shuffle=True,num_workers=8)
    print("Using",num_workers, "workers")

    computer = DNC()

    # load model:
    # computer, optim, starting_epoch, starting_iteration = load_model(computer)

    computer = computer.cuda()
    if optim is None:
        optimizer = torch.optim.Adam(computer.parameters(), lr=lr)
    else:
        # print('use Adadelta optimizer with learning rate ', lr)
        # optimizer = torch.optim.Adadelta(computer.parameters(), lr=lr)
        optimizer=optim

    real_criterion=nn.SmoothL1Loss()
    binary_criterion=nn.BCEWithLogitsLoss()

    # starting with the epoch after the loaded one

    train(computer, optimizer, real_criterion, binary_criterion,
          igdl, int(starting_epoch) + 1, total_epochs, iter_per_epoch, target_dim, logfile)

if __name__=="__main__":
    main()
