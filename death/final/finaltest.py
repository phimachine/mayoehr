# final test loads all models and run the test dataset

# Reviewed InputGen. I think this design works:
# For cn models, the model reads the whole sequence, and only the prediction at the last timestep will be evaluated
# For sequence models, the model reads the whole sequence, and the prediction loss will be averaged
# Besides loss, other metrics should be collected.

from death.post.inputgen_planH import InputGenH, pad_collate
import os
from os.path import abspath
from pathlib import Path

from death.DNC.seqtrainer import logprint, datetime_filename
from torch.utils.data import DataLoader
from death.final.losses import *
from death.final.metrics import *


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

    dnc=load_model(savestr[0],model_design_name[0],"DNC")[0]
    lstm=load_model(savestr[1],model_design_name[1],"baseline")[0]
    tacotron=load_model(savestr[2],model_design_name[2],"taco")[0]

    return dnc, tacotron, lstm

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
    sen=sensitivity(cause_of_death_output_sig,cause_of_death_target)
    spe=specificity(cause_of_death_output_sig,cause_of_death_target)
    f1=f1score(cause_of_death_output_sig,cause_of_death_target)
    prec=precision(cause_of_death_output_sig,cause_of_death_target)
    return loss, sen, spe, f1, prec


def loss_compare(savestr=("5toe","5toe","3toe"), model_design_name=("seqDNC", "lstmnorm", "taco")):

    logfile = "log/final_" + datetime_filename() + ".txt"

    num_workers = 16
    ig = InputGenH()
    validds = ig.get_valid()
    valid = DataLoader(dataset=validds, batch_size=8, num_workers=num_workers, collate_fn=pad_collate)
    valid_iterator=iter(valid)

    print("Using", num_workers, "workers for training set")
    dnc, tacotron, lstm = load_models(savestr=savestr,model_design_name=model_design_name)

    # load model:
    dnc = dnc.cuda()
    tacotron=tacotron.cuda()
    lstm=lstm.cuda()

    dnc_loss=0
    tacotron_loss=0
    lstm_loss=0

    dnc_metrics=[0]*4
    tacotron_metrics=[0]*4
    lstm_metrics=[0]*4

    val_batch=10
    for _ in range(val_batch):
        (input, target, loss_type) = next(valid_iterator)
        dl = run_one_patient(dnc, input, target, loss_type, 1e-5)
        tl = run_one_patient(tacotron,input,target,loss_type, 1e-5)
        ll = run_one_patient(lstm,input,target,loss_type, 1e-5)
        if dl is not None:
            dnc_loss += float(dl[0].data)
            tacotron_loss+=float(tl[0].data)
            lstm_loss+=float(ll[0].data)

            for i in range(4):
                dnc_metrics[i]+=dl[i]
                tacotron_metrics[i]+=tl[i]
                lstm_metrics[i]+=ll[i]
        else:
            raise ValueError("val_loss is none")

    dnc_loss = dnc_loss / val_batch
    tacotron_loss = tacotron_loss / val_batch
    lstm_loss = lstm_loss / val_batch
    logprint(logfile, "DNC. loss     : %.7f, sensitivity: %.5f, specificity: %.5f, f1: %5f, precision: %.5f" %
             (val_batch, dnc_loss, *dnc_metrics))
    logprint(logfile, "Tacotron. loss: %.7f, sensitivity: %.5f, specificity: %.5f, f1: %5f, precision: %.5f" %
             (val_batch, tacotron_loss, *tacotron_metrics))
    logprint(logfile, "LSTM. loss    : %.7f, sensitivity: %.5f, specificity: %.5f, f1: %5f, precision: %.5f" %
             (val_batch, lstm_loss, *lstm_metrics))



if __name__ == '__main__':
    loss_compare()