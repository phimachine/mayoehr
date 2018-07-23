import gc

import pandas as pd
import torch
import numpy
import pdb
from death.post.inputgen_planC import InputGen, train_valid_split
from torch.utils.data import DataLoader
import torch.nn as nn
from death.DNC.frankenstein import Frankenstein as DNC
from torch.autograd import Variable
import pickle
import traceback
import copy

batch_size = 1
debug=False

class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False

def copy_model_to_cpu(computer):
    model_dict=computer.state_dict()
    cpu_model_dict = {}
    for key, val in model_dict.items():
        cpu_model_dict[key] = val.cpu()

global_exception_counter = 0

def validate_batch_patients(cpu_computer, valid_iterator, target_dim, optimizer, real_criterion,
                            binary_criterion, valid_batch_num=10):
    optimizer.zero_grad()
    global global_exception_counter

    # this copies, right? You should check that by examining it after copying it back.
    # see if all weights are volatile

    running_loss=0
    computer=cpu_computer.cuda()

    for _ in range(valid_batch_num):
        if debug:
            print("validation")
        try:
            (input, target, loss_type) = next(valid_iterator)

            input = Variable(torch.Tensor(input).cuda(), volatile=True)
            target = Variable(torch.Tensor(target).cuda(), volatile=True)

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
                assert not (output != output).any()
                patient_output[0, timestep, :] = output

            # patient_output: (batch_size 1, time_length, output_dim ~4000)
            time_to_event_output = patient_output[:, :, 0]
            cause_of_death_output = patient_output[:, :, 1:]
            time_to_event_target = target[:, :, 0]
            cause_of_death_target = target[:, :, 1:]

            # this block will not work for batch input,
            # you should modify it so that the loss evaluation is not determined by logic but function.
            # def toe_loss_calc(real_criterion,time_to_event_output,time_to_event_target, patient_length):
            #
            # if loss_type[0] == 0:
            #     # in record
            #     toe_loss = real_criterion(time_to_event_output, time_to_event_target)
            #     cod_loss = binary_criterion(cause_of_death_output, cause_of_death_target)
            #     patient_loss = toe_loss/100 + cod_loss
            # else:
            #     # not in record
            #     # be careful with the sign, penalize when and only when positive
            #     underestimation = time_to_event_target - time_to_event_output
            #     underestimation = nn.functional.relu(underestimation)
            #     toe_loss = real_criterion(underestimation, torch.zeros_like(underestimation).cuda())
            #     cod_loss = binary_criterion(cause_of_death_output, cause_of_death_target)
            #     patient_loss = toe_loss/100 + cod_loss
            patient_loss = binary_criterion(cause_of_death_output, cause_of_death_target)

            running_loss += float(patient_loss[0])
            # del input, target, time_length, patient_output, timestep, feeding, output, \
            #     time_to_event_output, cause_of_death_output, time_to_event_target, cause_of_death_target, patient_loss
            # gc.collect()

        except ValueError:
            traceback.print_exc()
            print("Value Error reached")
            global_exception_counter += 1
            if global_exception_counter == 10:
                raise ValueError("Global exception counter reached 10. Likely the model has nan in memory")
            else:
                pass
    return running_loss


def train_one_patient(computer, input, target, target_dim, optimizer, loss_type, real_criterion,
                      binary_criterion):
    optimizer.zero_grad()
    global global_exception_counter
    try:
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
            assert not (output != output).any()
            patient_output[0, timestep, :] = output

        # patient_output: (batch_size 1, time_length, output_dim ~4000)
        time_to_event_output = patient_output[:, :, 0]
        cause_of_death_output = patient_output[:, :, 1:]
        time_to_event_target = target[:, :, 0]
        cause_of_death_target = target[:, :, 1:]

        # this block will not work for batch input,
        # you should modify it so that the loss evaluation is not determined by logic but function.
        # def toe_loss_calc(real_criterion,time_to_event_output,time_to_event_target, patient_length):
        #
        # if loss_type[0] == 0:
        #     # in record
        #     toe_loss = real_criterion(time_to_event_output, time_to_event_target)
        #     cod_loss = binary_criterion(cause_of_death_output, cause_of_death_target)
        #     patient_loss = toe_loss/100 + cod_loss
        # else:
        #     # not in record
        #     # be careful with the sign, penalize when and only when positive
        #     underestimation = time_to_event_target - time_to_event_output
        #     underestimation = nn.functional.relu(underestimation)
        #     toe_loss = real_criterion(underestimation, torch.zeros_like(underestimation).cuda())
        #     cod_loss = binary_criterion(cause_of_death_output, cause_of_death_target)
        #     patient_loss = toe_loss/100 + cod_loss
        patient_loss = binary_criterion(cause_of_death_output, cause_of_death_target)

        patient_loss.backward()
        optimizer.step()

        printloss = float(patient_loss[0])
        # del input, target, time_length, patient_output, timestep, feeding, output, \
        #     time_to_event_output, cause_of_death_output, time_to_event_target, cause_of_death_target, patient_loss
        # gc.collect()

    except ValueError:
        traceback.print_exc()
        print("Value Error reached")
        global_exception_counter += 1
        if global_exception_counter == 10:
            raise ValueError("Global exception counter reached 10. Likely the model has nan in memory")
        else:
            pass

    return printloss


def train(computer, optimizer, real_criterion, binary_criterion,
          train, valid_iterator, starting_epoch, total_epochs, starting_iter, iter_per_epoch, target_dim,
          logfile=False):
    print_interval = 10
    val_interval = 100
    save_interval = 200
    val_batch = 10
    if logfile:
        open(logfile, 'w').close()

    for epoch in range(starting_epoch, total_epochs):
        running_loss = 0
        for i, (input, target, loss_type) in enumerate(train):
            i = starting_iter + i
            if i < iter_per_epoch:
                printloss = train_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                                              real_criterion, binary_criterion)
                computer.new_sequence_reset()
                del input, target, loss_type
                running_loss += printloss
                if i % print_interval == 0:
                    running_loss = running_loss / print_interval
                    if i==0:
                        log_print("learning.   count: %4d, training loss: %.10f" %
                                  (i, printloss), logfile)
                    else:
                        log_print("learning.   count: %4d, training loss: %.10f, running loss: %.10f" %
                                  (i, printloss, running_loss), logfile)
                    # if logfile:
                    #     with open(logfile, 'a') as handle:
                    #         handle.write("learning.   count: %4d, training loss: %.10f \n" %
                    #                      (i, printloss))
                    #         if i!=0:
                    #             handle.write("count: %4d, running loss: %.10f \n" % (i, running_loss))
                    # print("learning.   count: %4d, training loss: %.10f" %
                    #       (i, printloss))
                    # if i!=0:
                    #     print("count: %4d, running loss: %.10f" % (i, running_loss))
                    running_loss = 0

                # every validation seems to produce some garbage in the GPU, why?
                if i % val_interval == 0:
                    if debug:
                        print("we have reached validation block.")
                    running_loss = 0
                    # cpu_computer=computer.cpu()
                    # copy_cpu_computer=copy.copy(cpu_computer)
                    # val_running_loss=validate_batch_patients(copy_cpu_computer, valid_iterator, target_dim, optimizer,
                    #                                          real_criterion, binary_criterion, val_batch)
                    cpu_computer=computer.cpu()
                    copy_cpu_computer=copy.deepcopy(cpu_computer)
                    val_running_loss=validate_batch_patients(copy_cpu_computer, valid_iterator, target_dim, optimizer,
                                                             real_criterion, binary_criterion, val_batch)
                    computer=cpu_computer.cuda()

                    # for i in range(val_interval):
                    #     if debug:
                    #         print("validation")
                    #     val_loss = train_one_patient(computer, input, target, target_dim, optimizer, loss_type,
                    #                                  real_criterion, binary_criterion, validate=True)
                    #     del input, target, loss_type
                    #     running_loss += val_loss
                    log_print("validation. count: %4d, val loss     : %.10f" %
                              (i, val_running_loss / val_batch), logfile)
                    #     with open(logfile, 'a') as handle:
                    #         handle.write("validation. count: %4d, val loss     : %.10f \n" %
                    #                      (i, printloss/val_batch))
                    # print("validation. count: %4d, training loss: %.10f" %
                    #       (i, printloss/val_batch))
                if i == 0:
                    save_model(computer, optimizer, epoch, i)
                    print("model saved for epoch", epoch, "iteration", i)
                else:
                    if i < save_interval:
                        if i % save_interval // 10 == save_interval // 10 - 1:
                            save_model(computer, optimizer, epoch, i)
                            print("model saved for epoch", epoch, "iteration", i)
                    if i > save_interval:
                        if i % save_interval // 2 == save_interval // 2 - 1:
                            save_model(computer, optimizer, epoch, i)
                            print("model saved for epoch", epoch, "iteration", i)
            else:
                break

def main():

    # this function is setup so that it tests validation and model saving first.
    total_epochs = 10
    iter_per_epoch = 100000
    lr = 1e-5
    target_dim = 3656

    logfile = "log.txt"

    num_workers = 3
    ig = InputGen()
    # multiprocessing needs to be careful, because socket request seems unstable.
    # if you request more workers than needed, you close and reopen sockets and the program might abort.
    # performance is best at 3
    trainds, validds = train_valid_split(ig, split_fold=10)
    traindl = DataLoader(dataset=trainds, batch_size=1, num_workers=num_workers)
    validdl = DataLoader(dataset=validds, batch_size=1)
    print("Using", num_workers, "workers for training set")

    computer = DNC().cuda()

    # load model:
    if True:
        print("loading model")
        computer, optim, starting_epoch, starting_iteration = load_model(computer)

    computer = computer.cuda()
    if optim is None:
        optimizer = torch.optim.Adam(computer.parameters(), lr=lr)
    else:
        # print('use Adadelta optimizer with learning rate ', lr)
        # optimizer = torch.optim.Adadelta(computer.parameters(), lr=lr)
        optimizer = optim

    real_criterion = nn.SmoothL1Loss()
    binary_criterion = nn.BCEWithLogitsLoss(size_average=False)

    # starting with the epoch after the loaded one

    train(computer, optimizer, real_criterion, binary_criterion,
          traindl, iter(validdl), int(starting_epoch) , total_epochs, int(starting_iteration) , iter_per_epoch,
          target_dim, logfile)


if __name__ == "__main__":
    main()
