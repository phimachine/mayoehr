from training.datagen import gendata
from archi.computer import Computer
import torch
import numpy

def train(model, criterion, optimizer,story_limit):

    for epoch in range(epochs_count):

        for batch in epoch_batches_count:
            input_data, target_output, seq_len, weights=gendata(batch_size,story_limit)

            for step in story_limit:

                # feed the batch into the machine
                # Expected input dimension: (150, 27)
                # output: (150,27)

                # TODO see whether the batch is truly independent: it should
                # usually batch does not interfere with each other's logic

                output=model()



if __name__=="__main__":

    batch_size=64
    story_limit=150
    epoch_batches_count=1024
    epochs_count=100
    lr=1e-3
    model=Computer()
    model.reset_parameters()
    criterion=torch.nn.modules.loss.CrossEntropyLoss()
    optimizer=torch.optim.Adam(lr=lr)