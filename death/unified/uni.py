# because we have discarded time-step based predictions, we can use a unified training script.
# this is especially necessary because the computation resources were limited.
# the design is simple. I want to pass a tuple of models and share a single input generator.
# will this work take some time? Sure. But not so much. Especially not so much when I start working.

import pandas as pd
import torch
from pathlib import Path
import os
from os.path import abspath
from death.post.inputgen_planJ import InputGenJ, pad_collate
from death.DNC.priorDNC import PriorDNC
from death.adnc.adnc import APDNC
from death.baseline.lstmtrainerG import lstmwrapperG
from death.taco.model import Tacotron
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import traceback
import datetime
from death.final.losses import TOELoss
from death.final.killtime import out_of_time
from death.final.metrics import ConfusionMatrixStats
import code
from death.analysis.expectedroc import get_death_code_proportion
from death.adnc.otheradnc import *


def logprint(logfile, string):
    string = str(string)
    if logfile is not None and logfile != False:
        with open(logfile, 'a') as handle:
            handle.write(string + '\n')
    print(string)

def file_exam(filename):
    with open(filename, "r") as handle:
        for line in handle:
            print(line)

def datetime_filename():
    return datetime.datetime.now().strftime("%m_%d_%X")

def sv(var):
    return var.data.cpu().numpy()

class ModelManager():
    def __init__(self, save_str="defuni", total_epochs=40, batch_size=64, beta=1e-6, num_workers=32, kill_time=False,
                 binary_criterion=nn.BCEWithLogitsLoss, time_criterion=TOELoss,
                 valid_batches=2048, moving_len=50):
        self.models=[]
        self.model_names=[]
        self.log_files=[]
        self.optims=[]
        self.total_epochs=total_epochs
        self.beta=beta
        self.kill_time=kill_time
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.save_str=save_str
        self.param_x=None
        self.param_v_t=None
        self.ig=None

        self.trainds=None
        self.validds=None
        self.traindl=None
        self.validdl=None
        self.prior_probability=None
        self.he=None
        self.hi=None
        self.optim=None

        self.binary_criterions=binary_criterion
        self.time_criterions=time_criterion

        self.moving_len=moving_len
        self.valid_batches=valid_batches

    def init_input_gen(self, inputgen_class, use_cache, collate_fn, *args, **kwargs):
        self.ig=inputgen_class(*args,**kwargs)
        self.param_x=self.ig.input_dim
        self.param_v_t=self.ig.output_dim
        if use_cache:
            self.trainds=self.ig.get_train_cached()
            self.validds=self.ig.get_valid_cached()
        else:
            self.trainds=self.ig.get_train()
            self.validds=self.ig.get_valid()
        self.traindl = DataLoader(dataset=self.trainds, batch_size=self.batch_size, num_workers=self.num_workers,
                                 collate_fn=collate_fn, pin_memory=True)
        self.validdl = DataLoader(dataset=self.validds, batch_size=self.batch_size, num_workers=self.num_workers // 2,
                                 collate_fn=collate_fn, pin_memory=True)

    def add_model(self, model, model_name):
        self.models.append(model)
        logfile = "log/" + model_name + "_" + datetime_filename() + ".txt"
        self.log_files.append(logfile)
        self.model_names.append(model_name)

    def init_optims_and_criterions(self, optim_class, lr):
        # you need to add optim after adding all models
        # can be modified to use a different optimizer for each model
        for model in self.models:
            self.optims.append(optim_class(model.parameters(),lr=lr))

        bc=self.binary_criterions
        tc=self.time_criterions

        self.binary_criterions=[]
        self.time_criterions=[]
        for i in range(len(self.models)):
            self.binary_criterions.append(bc())
            self.time_criterions.append(tc())

    def save_models(self, epoch, iteration):

        epoch = int(epoch)
        task_dir = os.path.dirname(abspath(__file__))
        if not os.path.isdir(Path(task_dir) / "saves" / self.save_str):
            os.mkdir(Path(task_dir) / "saves" / self.save_str)

        for model, model_name, optim in zip(self.models, self.model_names, self.optims):
            pickle_file = Path(task_dir).joinpath(
                "saves/" + self.save_str + "/"+model_name+"_" + str(epoch) + "_" + str(iteration) + ".pkl")
            with pickle_file.open('wb') as fhand:
                torch.save((model, optim, epoch, iteration), fhand)

            print("saved model",model_name,"at", pickle_file)


    def load_models(self):
        models=[]
        optims=[]
        hes=[]
        his=[]
        for model, name, optim in zip(self.models, self.model_names, self.optims):
            model, optim, highest_epoch, highest_iter = self.load_model(model, optim, 0, 0, self.save_str)
            models.append(model)
            optims.append(optim)
            hes.append(highest_epoch)
            his.append(highest_iter)

        he=hes[0]
        hi=his[0]
        for i in range(len(self.models)):
            assert(hes[i]==he)
            assert(his[i]==hi)

        self.models=models
        self.optims=optims
        print("All models loaded")
        self.he=he
        self.hi=hi

        return he, hi

    def load_model(self, computer, optim, starting_epoch, starting_iteration, savestr, model_name):
        task_dir = os.path.dirname(abspath(__file__))
        save_dir = Path(task_dir) / "saves" / savestr
        highestepoch = 0
        highestiter = 0

        if not os.path.isdir(Path(task_dir) / "saves" / savestr):
            os.mkdir(Path(task_dir) / "saves" / savestr)

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
            print("nothing to load")
            return computer, optim, starting_epoch, starting_iteration
        if starting_epoch == 0 and starting_iteration == 0:
            pickle_file = Path(task_dir).joinpath(
                "saves/" + savestr + "/"+model_name+"_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
        else:
            pickle_file = Path(task_dir).joinpath(
                "saves/" + savestr + "/"+model_name+"_" + str(starting_epoch) + "_" + str(starting_iteration) + ".pkl")
        print("loading model at", pickle_file)
        with pickle_file.open('rb') as pickle_file:
            computer, optim, epoch, iteration = torch.load(pickle_file)
        print('Loaded model at epoch ', highestepoch, 'iteration', iteration)

        return computer, optim, highestepoch, highestiter


    def train(self):
        if self.he is not None and self.hi is not None:
            starting_epoch, starting_iter = self.he, self.hi
        else:
            starting_epoch, starting_iter = 0, 0

        valid_iterator = iter(self.validdl)
        print_interval = 100
        val_interval = 999
        save_interval = 5
        val_batch = int(self.valid_batches / self.batch_size)

        cmss=[]
        for name in self.model_names:
            traincm, validcm = ConfusionMatrixStats(self.param_v_t- 1, string=name+" train"), ConfusionMatrixStats(self.param_v_t - 1, string=name+" valid")
            cmss.append((traincm, validcm))

        for computer, logfile in zip(self.models, self.log_files):
            for name, param in computer.named_parameters():
                logprint(logfile, name)
                logprint(logfile, param.data.shape)

        for epoch in range(starting_epoch, self.total_epochs):

            i=0

            if epoch % save_interval == 0:
                self.save_models(epoch, i)
                print("model saved for epoch", epoch, "iteration", i)

            for i, (input, target, loss_type) in enumerate(self.traindl):
                i = starting_iter + i
                if self.kill_time:
                    out_of_time()
                # train
                self.run_one_patient(i, input, target, loss_type, cmss, validate=False)
                if i % print_interval==0:
                    for model, name, logfile, cms in zip(self.models, self.model_names, self.log_files, cmss):
                        traincm, _ = cms[0], cms[1]
                        cod_loss, toe_loss=traincm.running_loss()
                        logprint(logfile, name+" epoch %4d, batch %4d. running cod: %.5f, toe: %.5f, total: %.5f" %
                                 (epoch, i, cod_loss, toe_loss, cod_loss + self.beta * toe_loss))
                        logprint(logfile, name+" train sen: %.6f, spe: %.6f, roc: %.6f" %
                                 tuple(traincm.running_stats()))

                # running validation before training might cause problem
                if i % val_interval == 0:
                    for _ in range(val_batch):
                        # we should consider running validation multiple times and average. TODO
                        try:
                            (input, target, loss_type) = next(valid_iterator)
                        except StopIteration:
                            valid_iterator = iter(self.validdl)
                            (input, target, loss_type) = next(valid_iterator)
                        self.run_one_patient(i, input, target, loss_type, cmss, validate=True)
                    for model, name, logfile, cms in zip(self.models, self.model_names, self.log_files, cmss):
                        _, validcm= cms[0], cms[1]
                        cod_loss, toe_loss=validcm.running_loss()
                        logprint(logfile, name + " validation. cod: %.10f, toe: %.10f, total: %.10f" %
                                 (cod_loss, toe_loss, cod_loss + self.beta * toe_loss))
                        logprint(logfile, name + " validate sen: %.6f, spe: %.6f, roc: %.6f" %
                                 tuple(validcm.running_stats()))

            starting_iter = 0

    def run_one_patient(self, index, input, target, loss_type, cmss, validate=False):
        for model, name, optim, logfile, cms, bin, time in zip(self.models, self.model_names, self.optims,
                                                               self.log_files, cmss, self.binary_criterions, self.time_criterions):
            self.run_one_patient_one_model(model, input, target, loss_type, optim, cms, bin, time, validate)

    def run_one_patient_one_model(self, computer, input, target, loss_type, optim, cms, binary, real, validate):
        global global_exception_counter
        patient_loss = None
        traincm = cms[0]
        validcm = cms[1]
        try:
            optim.zero_grad()

            input = Variable(torch.Tensor(input).cuda())
            target = Variable(torch.Tensor(target).cuda())
            loss_type = Variable(torch.Tensor(loss_type).cuda())

            # we have no critical index, becuase critical index are those timesteps that
            # DNC is required to produce outputs. This is not the case for our project.
            # criterion does not need to be reinitiated for every story, because we are not using a mask

            patient_output = computer(input)
            cause_of_death_output = patient_output[:, 1:]
            cause_of_death_target = target[:, 1:]
            # pdb.set_trace()
            cod_loss = binary(cause_of_death_output, cause_of_death_target)

            toe_output = patient_output[:, 0]
            toe_target = target[:, 0]
            toe_loss = real(toe_output, toe_target, loss_type)

            total_loss = cod_loss + self.beta* toe_loss
            sigoutput = torch.sigmoid(cause_of_death_output)

            # this numerical issue destroyed this model training.
            # loss keeps going down, but ROC is the same. I need to load an earlier epoch and restart.
            # I need to reimplement my BCE
            # this is a known issue for PyTorch 0.3.1 https://github.com/pytorch/pytorch/issues/2866

            # this happens when the logit is exactly 1 (\sigma(593)), and the loss is around 0.01
            # I decide to not debug it, because the backwards signal should still be usable.

            # no. the actual problem is that the death target does not conform one-hot requirement.
            # it's not the sparsity, because the output was one
            if cod_loss.data[0] < 0:
                import pickle
                # this was reached with loss negative. Why did that happen?
                # BCE loss is supposed to be positive all the time.
                with open("debug/itcc.pkl", 'wb') as f:
                    # this is a 1 Gb pickle. Somehow loading it takes forever. What?
                    pickle.dump((input, target, cause_of_death_output, cause_of_death_target), f)
                print(cod_loss.data[0])
                code.interact(local=locals())
                raise ValueError

            if not validate:
                total_loss.backward()
                optim.step()
                cod_loss=float(cod_loss.data)
                toe_loss=float(toe_loss.data)
                traincm.update_one_pass(sigoutput, cause_of_death_target, cod_loss, toe_loss)
                return cod_loss, toe_loss

            else:
                cod_loss=float(cod_loss.data)
                toe_loss=float(toe_loss.data)
                validcm.update_one_pass(sigoutput, cause_of_death_target, cod_loss, toe_loss)
                return cod_loss, toe_loss

            if global_exception_counter > -1:
                global_exception_counter -= 1
        except ValueError:
            traceback.print_exc()
            print("Value Error reached")
            print(datetime.datetime.now().time())
            global_exception_counter += 1
            if global_exception_counter == 10:
                save_model(computer, optimizer, epoch=0, iteration=global_exception_counter)
                raise ValueError("Global exception counter reached 10. Likely the model has nan in weights")
            else:
                raise


class ExperimentManager(ModelManager):
    def __init__(self, *args, **kwargs):
        super(ExperimentManager, self).__init__(*args, **kwargs)


    def initialize(self, load=False):
        self.init_input_gen(InputGenJ, collate_fn=pad_collate)
        self.add_DNC()
        self.add_LSTM()
        # self.add_Tacotron()
        if load:
            self.load_models()
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    def run(self):
        self.train()

    def add_DNC(self):
        param_h = 64  # 64
        param_L = 4  # 4
        param_W = 8  # 8
        param_R = 8  # 8
        param_N = 64  # 64
        param_x=self.param_x
        param_v_t=self.param_v_t

        prior_probability = get_death_code_proportion(self.ig)

        computer = PriorDNC(x=param_x,
                            h=param_h,
                            L=param_L,
                            v_t=param_v_t,
                            W=param_W,
                            R=param_R,
                            N=param_N,
                            prior=prior_probability)
        self.add_model(computer.cuda(), "priorDNC")

    def add_LSTM(self):
        # nothing to be changed
        lstm=lstmwrapperG(input_size=self.param_x, output_size=self.param_v_t)
        self.add_model(lstm.cuda(),"lstmG")

    def add_Tacotron(self):
        # add parameters in the hp.file
        taco=Tacotron(self.param_x, self.param_v_t)
        self.add_model(taco.cuda(),"taco")

    def add_APDNC(self):
        param_h = 64  # 64
        param_L = 4  # 4
        param_W = 8  # 8
        param_R = 8  # 8
        param_N = 64  # 64
        param_x=self.param_x
        param_v_t=self.param_v_t

        prior_probability = get_death_code_proportion(self.ig)

        apdnc=APDNC(x=param_x,
                    h=param_h,
                    L=param_L,
                    v_t=param_v_t,
                    W=param_W,
                    R=param_R,
                    N=param_N,
                    prior=prior_probability)

        self.add_model(apdnc.cuda(), "APDNC")

    def overfitting_experiment(self, load=False):
        self.save_str="overfitting"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        # small param DNC
        param_h = 16  # 64
        param_L = 2  # 4
        param_W = 4  # 8
        param_R = 4 # 8
        param_N = 16  # 64
        param_x=self.param_x
        param_v_t=self.param_v_t
        prior_probability = get_death_code_proportion(self.ig)


        dnc = PriorDNC(x=param_x,
                            h=param_h,
                            L=param_L,
                            v_t=param_v_t,
                            W=param_W,
                            R=param_R,
                            N=param_N,
                            prior=prior_probability)
        self.add_model(dnc.cuda(), "onefourthDNC")

        apdnc = APDNC(x=param_x,
                            h=param_h,
                            L=param_L,
                            v_t=param_v_t,
                            W=param_W,
                            R=param_R,
                            N=param_N,
                            prior=prior_probability)
        self.add_model(apdnc.cuda(), "onefourthADNC")

        if load:
            self.load_models()
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    def high_parameters(self,load=False):
        self.save_str = "highparam"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        param_h = 128  # 64
        param_L = 16  # 4
        param_W = 16  # 8
        param_R = 16  # 8
        param_N = 128  # 64
        param_x = self.param_x
        param_v_t = self.param_v_t
        prior_probability = get_death_code_proportion(self.ig)

        dnc = PriorDNC(x=param_x,
                       h=param_h,
                       L=param_L,
                       v_t=param_v_t,
                       W=param_W,
                       R=param_R,
                       N=param_N,
                       prior=prior_probability)
        self.add_model(dnc.cuda(), "doubleDNC")

        apdnc = APDNC(x=param_x,
                      h=param_h,
                      L=param_L,
                      v_t=param_v_t,
                      W=param_W,
                      R=param_R,
                      N=param_N,
                      prior=prior_probability)
        self.add_model(apdnc.cuda(), "doubleADNC")

        if load:
            self.load_models()
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)


    def adnc_exp(self, load=False):
        self.init_input_gen(InputGenJ, collate_fn=pad_collate)
        self.add_APDNC()

        if load:
            self.load_models()
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    def baseline(self,save_str="baseline", load=False):
        self.save_str=save_str
        self.init_input_gen(InputGenJ,collate_fn=pad_collate,use_cache=True)
        self.add_LSTM()
        self.add_Tacotron()
        if load:
            self.load_models()
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    def adnc_variations(self,load=False):

        self.save_str = "adncvariations"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        param_h =  64
        param_L = 4
        param_W = 8
        param_R = 8
        param_N = 64
        param_x = self.param_x
        param_v_t = self.param_v_t
        prior_probability = get_death_code_proportion(self.ig)

        dnc = ADNCNorm(x=param_x,
                       h=param_h,
                       L=param_L,
                       v_t=param_v_t,
                       W=param_W,
                       R=param_R,
                       N=param_N,
                       prior=prior_probability)
        self.add_model(dnc.cuda(), "ADNCNorm")

        dnc = ADNCbi(x=param_x,
                       h=param_h,
                       L=param_L,
                       v_t=param_v_t,
                       W=param_W,
                       R=param_R,
                       N=param_N,
                       prior=prior_probability)
        self.add_model(dnc.cuda(), "ADNCbi")

        dnc = ADNCDrop(x=param_x,
                       h=param_h,
                       L=param_L,
                       v_t=param_v_t,
                       W=param_W,
                       R=param_R,
                       N=param_N,
                       prior=prior_probability)
        self.add_model(dnc.cuda(), "ADNCDrop")


        dnc = ADNCMEM(x=param_x,
                       h=param_h,
                       L=param_L,
                       v_t=param_v_t,
                       W=param_W,
                       R=param_R,
                       N=param_N,
                       prior=prior_probability)
        self.add_model(dnc.cuda(), "ADNCMEM")

        if load:
            self.load_models()
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    def lstm_tacotron_with_prior(self, load=False):
        from death.baseline.priorlstm import PriorLSTM
        from death.taco.model import PriorTacotron

        self.save_str = "priorbaselines"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        prior_probability = get_death_code_proportion(self.ig)

        lstm=PriorLSTM(input_size=self.param_x, output_size=self.param_v_t, hidden_size=64, num_layers=4,
                       prior=prior_probability).cuda()
        taco=PriorTacotron(self.param_x, self.param_v_t, prior=prior_probability).cuda()

        self.add_model(lstm, "priorlstm")
        self.add_model(taco, "priortaco")

        if load:
            self.load_models()
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    def dnc_adnc_rerun(self,load=False):
        # log was incorrect, and validaion was incorrect. I want to run it again.

        self.save_str = "dnc_adnc_rerun"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        self.add_APDNC()
        self.add_DNC()

        if load:
            self.load_models()
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

def main():
    ds=ExperimentManager(batch_size=64, num_workers=8)
    ds.baseline()
    ds.run()


if __name__ == '__main__':
    main()