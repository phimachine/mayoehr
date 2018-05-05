from training.datagen import gendata
import torch
import numpy

def train(model, criterion, optimizer, ):


def train_one_batch(model,optimizer, criterion, input_sequence, price_targets):

    running_loss=0
    # input_sequence=input_sequence.float()
    output=model(input_sequence)

    # only outputs over time_length//4 will be considered
    # this has severe underflow problem.
    critical_output=output[time_length//5:]
    critical_price_targets=price_targets[time_length//5:]
    # critical_price_targets=critical_price_targets.float()
    running_loss += criterion(critical_output,critical_price_targets)

    running_loss.backward()
    optimizer.step()

    return running_loss.data[0]

def train(model, criterion, optimizer, total_batches):

    running_loss = 0.0

    for batch_num in range(total_batches):
        data=pruner.mass_batch_getter(time_length,batch_size)

        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        # inputs is a [64,128,9]
        # inputs=numpy.asarray(inputs)
        # labels=numpy.asarray(labels)
        # inputs=torch.from_numpy(inputs)
        # labels=torch.from_numpy(labels)
        # inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        #
        inputs=torch.from_numpy(inputs)
        labels=torch.from_numpy(labels)
        inputs=Variable(inputs).cuda()
        labels=Variable(labels).cuda()
        inputs=inputs.float()
        labels=labels.float()
        last_loss=train_one_batch(model,optimizer,criterion,inputs,labels)
        #
        #
        # # zero the parameter gradients
        # optimizer.zero_grad()
        #
        # # forward + backward + optimize
        # outputs = net(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # print statistics
        print_frequency=10
        save_frequency=1000
        running_loss += last_loss
        if batch_num % print_frequency == print_frequency-1:    # print every 2000 mini-batches
            print('[%5d] loss: %.3f' %
                  (batch_num + 1, running_loss / print_frequency))
            running_loss = 0.0
        if batch_num% save_frequency==save_frequency-1:
            print("Saving state dict")
            statedict_path=Path("trainer/state_dict_"+str(batch_num)+" MSE.pkl")
            torch.save(model.state_dict(),statedict_path)
    print("Finished training")


if __name__=="__main__":

    batch_size=64
    story_limit=150
    epoch_batches_count=1024
    epochs_count=100

    for epoch in range(epochs_count):

        for batch in epoch_batches_count:
            input_data, target_output, seq_len, weights=gendata(batch_size,story_limit)
