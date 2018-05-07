from training.datagen import gendata
from archi.computer import Computer
import torch
import numpy

# TODO weighted loss
# TODO save model parameters

def save_model():
    pass

def validate_one_batch_story(computer,criterion, optimizer, story_limit):
    input_data, target_output, seq_len, weights = gendata(batch_size, story_limit,validate=True)

    with torch.no_grad:

        story_loss = 0
        optimizer.zero_grad()

        # a single story
        for timestep in range(story_limit):
            # feed the batch into the machine
            # Expected input dimension: (150, 27)
            # output: (150,27)
            batch_input_of_same_timestep=input_data[:,timestep,:]
            batch_target_output_of_same_timestep=target_output[:,timestep,:]

            # usually batch does not interfere with each other's logic

            batch_output_of_same_timestep = computer(batch_input_of_same_timestep)
            story_loss += criterion(batch_output_of_same_timestep, batch_target_output_of_same_timestep)

        # I chose to backward a derivative only after a whole story has been taken in
        # This should lead to a more stable, but initially slower convergence.
        story_loss.backward()
        optimizer.step()
        computer.new_sequence_reset()

        # new pytorch
        return story_loss[0]


def train_one_batch_story(computer,criterion, optimizer, story_limit):
    input_data, target_output, seq_len, weights = gendata(batch_size, story_limit)

    story_loss = 0
    optimizer.zero_grad()

    # a single story
    for timestep in range(story_limit):
        # feed the batch into the machine
        # Expected input dimension: (150, 27)
        # output: (150,27)
        batch_input_of_same_timestep = input_data[:, timestep, :]
        batch_target_output_of_same_timestep = target_output[:, timestep, :]

        # usually batch does not interfere with each other's logic

        batch_output_of_same_timestep = computer(batch_input_of_same_timestep)
        story_loss += criterion(batch_output_of_same_timestep, batch_target_output_of_same_timestep)

    # I chose to backward a derivative only after a whole story has been taken in
    # This should lead to a more stable, but initially slower convergence.
    story_loss.backward()
    optimizer.step()
    computer.new_sequence_reset()

    # new pytorch
    return story_loss[0]

def train(computer, criterion, optimizer, story_limit):
    train_loss_history=[]
    test_history=[]
    for epoch in range(epochs_count):

        running_loss=0

        for batch in range(epoch_batches_count):
            # TODO this needs to be separated from training.

            train_story_loss=train_one_batch_story(computer,criterion.optimizer,story_limit)

            running_loss+=train_story_loss

            if batch%256==255:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch + 1, running_loss / 256))
                running_loss=0
                # also test the model
                val_loss=validate_one_batch_story(computer,criterion, optimizer, story_limit)
                print('[%d, %5d] loss: %.3f: Validation' %
                      (epoch + 1, batch + 1, val_loss))
                test_history+=[val_loss]

            train_loss_history+=[train_story_loss]



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