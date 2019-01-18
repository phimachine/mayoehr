# because loading all models together causes out-of-memory error, I need to load them and test them one by one
# all of the metrics take into consideration structural codes.
# For code A1234, the machine needs to predict all.

from death.post.inputgen_planH import InputGenH, pad_collate
import os
from os.path import abspath
from pathlib import Path

from death.DNC.seqtrainer import logprint, datetime_filename
from torch.utils.data import DataLoader
from death.final.losses import *
from death.final.metrics import *

import torch

def load_models(savestr, model_design_name):
    """
    Similar to load_model() used in trainer files, but not exactly the same
    It assumes that the load will be successful and does not initiate an empty model
    It loads multiple files
    :return:
    """
    def load_model(savestr, model_design_name, folder_name):
        task_dir = os.path.dirname(abspath(__file__))
        save_dir = Path(task_dir).parent/ folder_name / "saves" / savestr
        highestepoch = 0
        highestiter = 0
        for child in save_dir.iterdir():
            try:
                epoch = str(child).split("_")[3]
                iteration = str(child).split("_")[4].split('.')[0]
            except IndexError:
                print(str(child))
            iteration = int(iteration)
            epoch = int(epoch)
            # some files are open but not written to yet.
            if child.stat().st_size > 20480:
                if epoch > highestepoch or (iteration > highestiter and epoch == highestepoch):
                    highestepoch = epoch
                    highestiter = iteration
        if highestepoch == 0 and highestiter == 0:
            raise FileNotFoundError("nothing to load")
        pickle_file = save_dir / (model_design_name+"_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
        print("loading model at", pickle_file)
        with pickle_file.open('rb') as pickle_file:
            computer, optim, epoch, iteration = torch.load(pickle_file)
        print('Loaded model at epoch ', highestepoch, 'iteration', iteration)

        return computer, optim, highestepoch, highestiter

    # need to make sure the outer scope releases the memory
    yield load_model(savestr[0],model_design_name[0],"DNC")[0]
    yield load_model(savestr[1],model_design_name[1],"baseline")[0]
    yield load_model(savestr[2],model_design_name[2],"taco")[0]


def run_one_patient(computer, input, target, loss_type, beta):
    """
    Similar to training run_one_patient, with a cleaned up interface, and not expected to fail.
    :param computer:
    :param input:
    :param target:
    :param real_criterion:
    :param binary_criterion:
    :return:


    """
    binary_criterion = nn.BCEWithLogitsLoss()

    computer.eval()
    input = Variable(input.cuda())
    target = Variable(target.cuda())
    loss_type = Variable(loss_type.float()).cuda()
    # we have no critical index, becuase critical index are those timesteps that
    # DNC is required to produce outputs. This is not the case for our project.
    # criterion does not need to be reinitiated for every story, because we are not using a mask

    patient_output = computer(input)
    cause_of_death_output = patient_output[:, 1:]
    toe_output=patient_output[:,0]
    cause_of_death_target = target[:, 1:]
    toe_target=target[:,0]
    # pdb.set_trace()
    patient_loss = binary_criterion(cause_of_death_output, cause_of_death_target)
    TOE = TOELoss()
    toe_loss=TOE(toe_output,toe_target,loss_type)
    loss=patient_loss+beta*toe_loss

    # metrics
    cause_of_death_output_sig=torch.nn.functional.sigmoid(cause_of_death_output)
    # sen=sensitivity(cause_of_death_output_sig,cause_of_death_target)
    # spe=specificity(cause_of_death_output_sig,cause_of_death_target)
    # f1=f1score(cause_of_death_output_sig,cause_of_death_target)
    # prec=precision(cause_of_death_output_sig,cause_of_death_target)
    return float(loss.data), cause_of_death_output_sig.data, cause_of_death_target.data



def loss_compare(savestr=("poswei","maxpool","poswei"), model_design_name=("seqDNC", "lstmnorm", "taco")):
    logfile = "log/final_" + datetime_filename() + ".txt"
    models = load_models(savestr=savestr, model_design_name=model_design_name)

    for mdn in model_design_name:
        bs=8
        num_workers = 8
        small_target=True
        ig = InputGenH(small_target=small_target)
        if small_target:
            outputlen=2975
        else:
            outputlen=5951
        # use validation for the moment
        testds = ig.get_test()
        test = DataLoader(dataset=testds, batch_size=bs, num_workers=num_workers, collate_fn=pad_collate)
        valid_iterator=iter(test)
        model=next(models)
        model=model.cuda()
        loss=0

        val_batch=25
        oo=torch.zeros((val_batch*bs,outputlen))
        tt=torch.zeros((val_batch*bs,outputlen))
        for i in range(val_batch):
            (input, target, loss_type) = next(valid_iterator)
            dl = run_one_patient(model, input, target, loss_type, 1e-5)
            if dl is not None:
                loss += dl[0]

                oo[i*8:i*8+8,:]=dl[1]
                tt[i*8:i*8+8,:]=dl[2]
            else:
                raise ValueError("val_loss is none")
        loss=loss/val_batch
        # averaging the metrics is not the correct approach.
        # we need to concatenate all results to calculate the metrics.
        sen=sensitivity(oo,tt)
        spe=specificity(oo,tt)
        f1=f1score(oo,tt)
        prec=precision(oo,tt)
        acc=accuracy(oo,tt)
        logprint(logfile, "%s. loss     : %.7f, sensitivity: %.5f, specificity: %.5f, precision: %.5f, f1: %.5f, accuracy: %.5f" %
                 (mdn, loss, sen, spe, prec, f1, acc))
        # del model


if __name__ == '__main__':
    loss_compare()

# TODO partial matching is not allowed. You need to slice the codes to perfect matching only.