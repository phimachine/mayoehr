from training.datagen import gendata
from archi.computer import Computer
import torch
import numpy

def train(computer, criterion, optimizer, story_limit):

    for epoch in range(epochs_count):

        for batch in range(epoch_batches_count):
            # TODO this needs to be separated from training.
            input_data, target_output, seq_len, weights=gendata(batch_size,story_limit)

            for timestep in story_limit:
                running_loss = 0
                # feed the batch into the machine
                # Expected input dimension: (150, 27)
                # output: (150,27)

                # TODO see whether the batch is truly independent: it should
                # usually batch does not interfere with each other's logic

                # slice data

                batch_output_of_same_timestep=computer(batch_input_of_same_timestep)
                running_loss+=criterion(batch_output_of_same_timestep,batch_target_output_of_same_timestep)
                running_loss.backward()
                optimizer.step()
            computer.new_sequence_reset()



if __name__=="__main__":

    batch_size=64
    story_limit=150
    epoch_batches_count=1024
    epochs_count=100
    lr=1e-3
    computer=Computer()
    computer.reset_parameters()
    criterion=torch.nn.modules.loss.CrossEntropyLoss()
    optimizer=torch.optim.Adam(lr=lr)

    train(computer,criterion,optimizer,story_limit)