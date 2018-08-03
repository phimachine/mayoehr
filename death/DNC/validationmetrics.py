import torch
import numpy as np
from pathlib import Path
import os
from os.path import abspath
from torch.autograd import Variable
from torch.utils.data import DataLoader
from death.DNC.trashcan.trainerD import InputGenD, train_valid_split

def sensitivity(target, output):
    '''

    :param target: np array of bool
    :param output: np array of float
    :return:
    '''

    # target is one

    truepositive=np.sum(target*output)
    positive=np.sum(target)
    if positive<1e-6:
        positive=1e-6

    return truepositive/positive

def specificity(target, output):
    truenegative=np.sum((1-target)*(1-output))
    negative=np.sum(1-target)
    if negative<1e-6:
        negative=1e-6

    return truenegative/negative

def precision(target,output):
    truepositive=np.sum(output*target)
    yes=np.sum(output)
    if yes<1e-6:
        yes=1e-6
    return truepositive/yes

def recall(target,output):
    return sensitivity(target,output)

def f1score(target,output):
    rec=recall(target,output)
    if rec<1e-6:
        rec=1e-6
    return precision(target,output) / rec

def evaluate_one_patient(computer,input,target,target_dim):
    input = Variable(torch.Tensor(input).cuda())
    time_length = input.size()[1]
    # with torch.no_grad if validate else dummy_context_mgr():
    patient_output = Variable(torch.Tensor(1, time_length, target_dim)).cuda()
    for timestep in range(time_length):
        # first colon is always size 1
        feeding = input[:, timestep, :]
        output = computer(feeding)
        assert not (output != output).any()
        patient_output[0, timestep, :] = output

    time_to_event_output = patient_output[:, :, 0]
    cause_of_death_output = patient_output[:, :, 1:]
    time_to_event_target = target[:, :, 0]
    cause_of_death_target = target[:, :, 1:]

    cause_of_death_output=cause_of_death_output.data.cpu().numpy()
    cause_of_death_target=cause_of_death_target.cpu().numpy()
    cause_of_death_output=1/(1+np.exp(-cause_of_death_output))

    assert (cause_of_death_target-1<1e-6).all()
    assert (cause_of_death_output-1<1e-6).all()
    assert (cause_of_death_output>-1e-6).all()
    assert (cause_of_death_target>-1e-6).all()

    sen=sensitivity(cause_of_death_target,cause_of_death_output)
    spe=specificity(cause_of_death_target,cause_of_death_output)
    f1=f1score(cause_of_death_target,cause_of_death_output)
    prec=precision(cause_of_death_target,cause_of_death_output)

    return sen,spe,f1, prec

def evaluatemodel():
    task_dir = os.path.dirname(abspath(__file__))

    pickle_file = Path(task_dir).joinpath("saves/DNCfull_0_4900.pkl")
    print("loading model at", pickle_file)
    pickle_file = pickle_file.open('rb')
    computer, optim, epoch, iteration = torch.load(pickle_file)

    num_workers = 8
    ig = InputGenD()
    trainds,validds=train_valid_split(ig,split_fold=10)
    traindl = DataLoader(dataset=trainds, batch_size=1, num_workers=num_workers)
    validdl = DataLoader(dataset=validds, batch_size=1)
    print("Using", num_workers, "workers for training set")
    target_dim=None

    running_sensitivity=[]
    running_specificity=[]
    running_f1=[]
    running_precision=[]


    for i, (input, target, loss_type) in enumerate(traindl):
        if target_dim is None:
            target_dim = target.shape[2]

        if i<10:
            sen,spe,f1,prec=evaluate_one_patient(computer,input,target,target_dim)
            running_sensitivity.append(sen)
            running_specificity.append(spe)
            running_f1.append(f1)
            running_precision.append(prec)
        else:
            break
        print("next")

    print("average stats of 10 patients:")
    print("sensitivity: ",np.mean(running_sensitivity))
    print("specificity: ", np.mean(running_specificity))
    print("f1: ",np.mean(running_f1))
    print("precision: ",np.mean(running_precision))

def smalltest():
    target=np.array([1,1,1,1,0,0],dtype=bool)
    output=np.array([0,1,1,1,1,0],dtype=bool)
    # sensitivity: 75%
    # specificity: 50%
    # precision: 75%

    print(sensitivity(target,output))
    print(specificity(target,output))
    print(precision(target,output))


if __name__=="__main__":
    evaluatemodel()