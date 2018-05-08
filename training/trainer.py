from training.datagen import gendata
from archi.computer import Computer
import torch
import numpy
import archi.param as param

# task 10 of babi

batch_size=param.bs
word_space=27
param.x=word_space
param.v_t=word_space

class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

# TODO save model parameters

def save_model():
    pass

def run_one_story(computer, optimizer, story_length, batch_size, validate=False):
    # to promote code reuse
    input_data, target_output, seq_len, weights = gendata(batch_size, validate=validate)
    input_data=torch.Tensor(input_data)
    target_output=torch.Tensor(target_output)
    weights=torch.Tensor(weights)

    criterion=torch.nn.CrossEntropyLoss(weight=weights)

    with torch.no_grad if validate else dummy_context_mgr():

        story_output = torch.Tensor(batch_size, story_length,word_space)

        # a single story
        for timestep in range(story_length):
            # feed the batch into the machine
            # Expected input dimension: (150, 27)
            # output: (150,27)
            batch_input_of_same_timestep = input_data[:, timestep, :]

            # usually batch does not interfere with each other's logic
            batch_output=computer(batch_input_of_same_timestep)
            story_output[:, timestep,:] = batch_output
        # shouldn't have made it one hot, but let's do some redundant work here
        _,target_output=target_output.max(dim=2)
        story_loss = criterion(story_output, target_output)
        if not validate:
            # I chose to backward a derivative only after a whole story has been taken in
            # This should lead to a more stable, but initially slower convergence.
            story_loss.backward()
            optimizer.step()
        computer.new_sequence_reset()

    # new pytorch support
    return story_loss[0]

#
# def validate_one_batch_story(computer, story_length):
#     input_data, target_output, seq_len, weights = gendata(batch_size, validate=True)
#     criterion=torch.nn.CrossEntropyLoss(weights=weights)
#
#     with torch.no_grad:
#
#         story_output = torch.Tensor(batch_size, story_length)
#
#         # a single story
#         for timestep in range(story_length):
#             # feed the batch into the machine
#             # Expected input dimension: (150, 27)
#             # output: (150,27)
#             batch_input_of_same_timestep = input_data[:, timestep, :]
#
#             # usually batch does not interfere with each other's logic
#             story_output[:, timestep] = computer(batch_input_of_same_timestep)
#
#         story_loss = criterion(story_output, target_output)
#         computer.new_sequence_reset()
#
#         # new pytorch support
#         return story_loss[0]
#
#
# def train_one_batch_story(computer, optimizer, story_length, batch_size):
#     input_data, target_output, seq_len, weights = gendata(batch_size, story_length)
#     weights=torch.Tensor(weights)
#     criterion=torch.nn.CrossEntropyLoss(weight=weights)
#
#     optimizer.zero_grad()
#
#     story_output=torch.Tensor(batch_size, story_length)
#
#     # a single story
#     for timestep in range(story_length):
#         # feed the batch into the machine
#         # Expected input dimension: (150, 27)
#         # output: (150,27)
#         batch_input_of_same_timestep = input_data[:, timestep, :]
#
#         # usually batch does not interfere with each other's logic
#         story_output[:, timestep] = computer(batch_input_of_same_timestep)
#
#
#     story_loss=criterion(story_output,target_output)
#     # I chose to backward a derivative only after a whole story has been taken in
#     # This should lead to a more stable, but initially slower convergence.
#     story_loss.backward()
#     optimizer.step()
#     computer.new_sequence_reset()
#
#     # new pytorch support
#     return story_loss[0]

def train(computer, optimizer, story_length, batch_size):
    train_loss_history=[]
    test_history=[]
    for epoch in range(epochs_count):

        running_loss=0

        for batch in range(epoch_batches_count):
            # TODO this needs to be separated from training.

            train_story_loss=run_one_story(computer, optimizer, story_length, batch_size)

            running_loss+=train_story_loss

            if batch%256==255:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch + 1, running_loss / 256))
                running_loss=0
                # also test the model
                val_loss=run_one_story(computer, optimizer, story_length, batch_size, validate=False)
                print('[%d, %5d] loss: %.3f: Validation' %
                      (epoch + 1, batch + 1, val_loss))
                test_history+=[val_loss]

            train_loss_history+=[train_story_loss]



if __name__=="__main__":

    story_limit=150
    epoch_batches_count=1024
    epochs_count=100
    lr=1e-3
    computer=Computer()
    computer.reset_parameters()
    optimizer=torch.optim.Adam(computer.parameters(),lr=lr)

    train(computer,optimizer,story_limit, batch_size)