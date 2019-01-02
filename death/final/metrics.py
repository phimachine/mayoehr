import torch
import numpy as np

def sensitivity(output, target):
    '''

    :param target: np array of bool
    :param output: np array of float
    :return:
    '''

    # target is one

    truepositive=torch.sum(target*output)
    real_positive=torch.sum(target)
    if real_positive<1e-6:
        real_positive=1e-6

    return (truepositive/real_positive)

def specificity(output, target):
    truenegative=torch.sum((1-target)*(1-output))
    real_negative=torch.sum(1-target)
    if real_negative<1e-6:
        real_negative=1e-6

    return (truenegative/real_negative)

def precision(output, target):
    truepositive=torch.sum(output*target)
    positive=torch.sum(output)
    if positive<1e-6:
        positive=1e-6
    return (truepositive/positive)

def recall(output, target):
    return sensitivity(output,target)

def f1score(output, target):
    # lol where did I get this formula before?
    rec=recall(output,target)
    if rec<1e-6:
        rec=1e-6
    prec=precision(output,target)
    if prec<1e-6:
        prec=1e-6
    f1=1/((1/rec+1/prec)/2)
    return f1

def accuracy(output,target):
    truepositive=torch.sum(output*target)
    truenegative=torch.sum((1-target)*(1-output))
    inc=target.nelement()

    return (truenegative+truepositive)/inc

#
# def evaluate_one_patient(computer,input,target,target_dim):
#     input = Variable(torch.Tensor(input).cuda())
#     time_length = input.size()[1]
#     # with torch.no_grad if validate else dummy_context_mgr():
#     patient_output = Variable(torch.Tensor(1, time_length, target_dim)).cuda()
#     for timestep in range(time_length):
#         # first colon is always size 1
#         feeding = input[:, timestep, :]
#         output = computer(feeding)
#         assert not (output != output).any()
#         patient_output[0, timestep, :] = output
#
#     time_to_event_output = patient_output[:, :, 0]
#     cause_of_death_output = patient_output[:, :, 1:]
#     time_to_event_target = target[:, :, 0]
#     cause_of_death_target = target[:, :, 1:]
#
#     cause_of_death_output=cause_of_death_output.data.cpu().numpy()
#     cause_of_death_target=cause_of_death_target.cpu().numpy()
#     cause_of_death_output=1/(1+torch.exp(-cause_of_death_output))
#
#     assert (cause_of_death_target-1<1e-6).all()
#     assert (cause_of_death_output-1<1e-6).all()
#     assert (cause_of_death_output>-1e-6).all()
#     assert (cause_of_death_target>-1e-6).all()
#
#     sen=sensitivity(cause_of_death_target,cause_of_death_output)
#     spe=specificity(cause_of_death_target,cause_of_death_output)
#     f1=f1score(cause_of_death_target,cause_of_death_output)
#     prec=precision(cause_of_death_target,cause_of_death_output)

    # return sen,spe,f1, prec
#
# def evaluatemodel():
#     task_dir = os.path.dirname(abspath(__file__))
#
#     pickle_file = Path(task_dir).joinpath("saves/DNCfull_0_4900.pkl")
#     print("loading model at", pickle_file)
#     pickle_file = pickle_file.open('rb')
#     computer, optim, epoch, iteration = torch.load(pickle_file)
#
#     num_workers = 8
#     ig = InputGenD()
#     trainds,validds=train_valid_split(ig,split_fold=10)
#     traindl = DataLoader(dataset=trainds, batch_size=1, num_workers=num_workers)
#     validdl = DataLoader(dataset=validds, batch_size=1)
#     print("Using", num_workers, "workers for training set")
#     target_dim=None
#
#     running_sensitivity=[]
#     running_specificity=[]
#     running_f1=[]
#     running_precision=[]
#
#
#     for i, (input, target, loss_type) in enumerate(traindl):
#         if target_dim is None:
#             target_dim = target.shape[2]
#
#         if i<10:
#             sen,spe,f1,prec=evaluate_one_patient(computer,input,target,target_dim)
#             running_sensitivity.append(sen)
#             running_specificity.append(spe)
#             running_f1.append(f1)
#             running_precision.append(prec)
#         else:
#             break
#         print("next")
#
#     print("average stats of 10 patients:")
#     print("sensitivity: ",torch.mean(running_sensitivity))
#     print("specificity: ", torch.mean(running_specificity))
#     print("f1: ",torch.mean(running_f1))
#     print("precision: ",torch.mean(running_precision))

def smalltest():
    target=torch.array([1,1,1,1,0,0],dtype=bool)
    output=torch.array([0,1,1,1,1,0],dtype=bool)
    # sensitivity: 75%
    # specificity: 50%
    # precision: 75%

    print(sensitivity(target,output))
    print(specificity(target,output))
    print(precision(target,output))


if __name__=="__main__":
    evaluatemodel()