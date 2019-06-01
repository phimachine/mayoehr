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
from death.lstmbaseline.priorlstm import PriorLSTM
from death.lstmbaseline.lstmtrainerG import lstmwrapperG
from death.taco.model import Tacotron
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
import datetime
from death.final.losses import TOELoss, DiscreteCrossEntropy
from death.final.killtime import out_of_time
from death.final.metrics import ConfusionMatrixStats
import code
from death.analysis.expectedroc import get_death_code_proportion
from death.adnc.otheradnc import *
from death.ehrtran.attnmodel import TransformerMixedAttn
from death.ehrtran.forwardmodel import TransformerMixedForward
from death.ehrtran.softmaxmodels import TransformerMixedForwardSoftmax, TransformerMixedAttnSoftmax
from death.taco.model import PriorTacotron
from death.ehrtran.simple import Simple

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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


def save_for_debugging(model, input_tuple):
    """
    Sometimes debugging frame is not available, you can use this function to take the whole model out together with the problem input
    :return:
    """
    with open("debug/save.pkl", 'wb') as f:
        torch.save((model, (input_tuple)), f)


def load_for_debugging(fpath="debug/save.pkl"):
    with open(fpath, 'rb') as f:
        model, (input_tuple) = torch.load(f)
    return model(*input_tuple)


class ModelManager():
    def __init__(self, save_str="defuni", total_epochs=40, batch_size=64, beta=1e-8, num_workers=4, kill_time=False,
                 binary_criterion=nn.BCEWithLogitsLoss, time_criterion=TOELoss,
                 valid_batches=2048, moving_len=50):
        self.models = []
        self.model_names = []

        self.transformer = []
        self.log_files = []
        self.optims = []
        self.total_epochs = total_epochs
        self.beta = beta
        self.kill_time = kill_time
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_str = save_str
        # 7298
        self.param_x = None
        # 435
        self.param_v_t = None
        self.ig = None

        self.trainds = None
        self.validds = None
        self.traindl = None
        self.validdl = None
        self.testds = None
        self.testdl = None
        self.prior_probability = None
        self.he = None
        self.hi = None
        self.optim = None

        self.binary_criterions = binary_criterion
        self.time_criterions = time_criterion

        self.moving_len = moving_len
        self.valid_batches = valid_batches

    def init_input_gen(self, inputgen_class, use_cache, collate_fn, *args, **kwargs):
        self.ig = inputgen_class(*args, **kwargs, cached=use_cache)
        self.param_x = self.ig.input_dim
        self.param_v_t = self.ig.output_dim
        if use_cache:
            self.trainds = self.ig.get_train_cached()
            self.validds = self.ig.get_valid_cached()
            self.testds = self.ig.get_test_cached()
        else:
            self.trainds = self.ig.get_train()
            self.validds = self.ig.get_valid()
            self.testds = self.ig.get_test()
        self.traindl = DataLoader(dataset=self.trainds, batch_size=self.batch_size, num_workers=self.num_workers,
                                  collate_fn=collate_fn, shuffle=True, pin_memory=True)
        self.validdl = DataLoader(dataset=self.validds, batch_size=self.batch_size, num_workers=self.num_workers // 2,
                                  collate_fn=collate_fn, shuffle=True, pin_memory=True)
        self.testdl = DataLoader(dataset=self.testds, batch_size=self.batch_size, num_workers=self.num_workers,
                                 collate_fn=collate_fn, shuffle=True, pin_memory=True)

    def add_model(self, model, model_name, transformer=False):
        """

        :param model:
        :param model_name:
        :param transformer: specify transformer, or anything else, to use different pipelines
        :return:
        """
        self.models.append(model)
        logfile = "log/" + self.save_str+ "_" + model_name + "_" + datetime_filename() + ".txt"
        self.log_files.append(logfile)
        self.model_names.append(model_name)
        self.transformer.append(transformer)

    def init_optims_and_criterions(self, optim_class, lr):
        # you need to add optim after adding all models
        # can be modified to use a different optimizer for each model
        for model in self.models:
            self.optims.append(optim_class(filter(lambda p: p.requires_grad, model.parameters()), lr=lr))

        bc = self.binary_criterions
        tc = self.time_criterions

        self.binary_criterions = []
        self.time_criterions = []
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
                "saves/" + self.save_str + "/" + model_name + "_" + str(epoch) + "_" + str(iteration) + ".pkl")
            with pickle_file.open('wb') as fhand:
                torch.save((model, optim, epoch, iteration), fhand)

            print("saved model", model_name, "at", pickle_file)

    def load_models(self, epoch=0):
        """

        :param epoch: 0 is load the newest, or load the epoch epoch.
        :return:
        """
        models = []
        optims = []
        hes = []
        his = []

        if len(self.optims) == 0:
            raise ValueError("Please initialize models and optims first")

        for model, name, optim in zip(self.models, self.model_names, self.optims):
            model, optim, highest_epoch, highest_iter = self.load_model(model, optim, epoch, 0, self.save_str, name)
            models.append(model)
            optims.append(optim)
            hes.append(highest_epoch)
            his.append(highest_iter)
        try:
            he = hes[0]
            hi = his[0]
        except:
            raise ValueError("No model to load")
        for i in range(len(self.models)):
            assert (hes[i] == he)
            assert (his[i] == hi)

        self.models = models
        self.optims = optims
        print("All models loaded")
        self.he = he
        self.hi = hi

        return he, hi

    def load_model(self, computer, optim, starting_epoch, starting_iteration, savestr, model_name):
        task_dir = os.path.dirname(abspath(__file__))
        save_dir = Path(task_dir) / "saves" / savestr
        highestepoch = 0
        highestiter = 0

        if not os.path.isdir(Path(task_dir) / "saves" / savestr):
            os.mkdir(Path(task_dir) / "saves" / savestr)

        for child in save_dir.iterdir():
            if child.name.split("_")[0]==model_name:
                try:
                    epoch = child.name.split("_")[1]
                    iteration = child.name.split("_")[2].split('.')[0]
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
                "saves/" + savestr + "/" + model_name + "_" + str(highestepoch) + "_" + str(highestiter) + ".pkl")
            print("loading model at", pickle_file)
            with pickle_file.open('rb') as pickle_file:
                computer, optim, epoch, iteration = torch.load(pickle_file)
            print('Loaded model at epoch ', highestepoch, 'iteration', highestiter)
        else:
            pickle_file = Path(task_dir).joinpath(
                "saves/" + savestr + "/" + model_name + "_" + str(starting_epoch) + "_" + str(starting_iteration) + ".pkl")
            print("loading model at", pickle_file)
            with pickle_file.open('rb') as pickle_file:
                computer, optim, epoch, iteration = torch.load(pickle_file)
            print('Loaded model at epoch ', starting_epoch, 'iteration', starting_iteration)

        return computer, optim, highestepoch, highestiter

    def train(self, verbose=True):
        if self.he is not None and self.hi is not None:
            starting_epoch, starting_iter = self.he, self.hi
        else:
            starting_epoch, starting_iter = 0, 0

        valid_iterator = iter(self.validdl)
        print_interval = 100
        val_interval = 500
        save_interval = 2
        val_batch = int(self.valid_batches / self.batch_size)

        cmss = []
        for name in self.model_names:
            traincm, validcm = ConfusionMatrixStats(self.param_v_t - 1, string=name + " train"), ConfusionMatrixStats(
                self.param_v_t - 1, string=name + " valid")
            cmss.append((traincm, validcm))

        if verbose:
            for computer, logfile in zip(self.models, self.log_files):
                for name, param in computer.named_parameters():
                    logprint(logfile, name)
                    logprint(logfile, param.data.shape)

        for epoch in range(starting_epoch, self.total_epochs):
            i = 0
            if epoch % save_interval == 0:
                self.save_models(epoch, i)
                print("model saved for epoch", epoch, "iteration", i)

            for i, (input, target, loss_type, time_length) in enumerate(self.traindl):
                i = starting_iter + i
                if self.kill_time:
                    out_of_time()
                # train
                self.run_one_patient(input, target, loss_type, time_length, cmss)
                if i % print_interval == 0:
                    for model, name, logfile, cms in zip(self.models, self.model_names, self.log_files, cmss):
                        traincm, _ = cms[0], cms[1]
                        cod_loss, toe_loss = traincm.running_loss()
                        logprint(logfile,
                                 "%14s " % name + "train epoch %4d, batch %4d. running cod: %.5f, toe: %.5f, total: %.5f" %
                                 (epoch, i, cod_loss, toe_loss, cod_loss + self.beta * toe_loss))
                        logprint(logfile, "%14s " % name + "train sen: %.6f, spe: %.6f, roc: %.6f" % tuple(
                            traincm.running_stats()))

                # running validation before training might cause problem
                if i % val_interval == 0:
                    for _ in range(val_batch):
                        # we should consider running validation multiple times and average. TODO
                        try:
                            (input, target, loss_type, time_length) = next(valid_iterator)
                        except StopIteration:
                            valid_iterator = iter(self.validdl)
                            (input, target, loss_type, time_length) = next(valid_iterator)
                        self.run_one_patient(input, target, loss_type, time_length, cmss, validate=True)
                    for model, name, logfile, cms in zip(self.models, self.model_names, self.log_files, cmss):
                        _, validcm = cms[0], cms[1]
                        cod_loss, toe_loss = validcm.running_loss()
                        logprint(logfile, "%14s " % name + "validation. cod: %.10f, toe: %.10f, total: %.10f" %
                                 (cod_loss, toe_loss, cod_loss + self.beta * toe_loss))
                        logprint(logfile, "%14s " % name + "validate sen: %.6f, spe: %.6f, roc: %.6f" % tuple(
                            validcm.running_stats()))

            starting_iter = 0

    def test(self, AUROC_alpha=0.001, verbose=False, test_stat_dir=None):

        print_interval = 100

        cmss = []
        for name in self.model_names:
            testcm = ConfusionMatrixStats(self.param_v_t - 1, string=name + " test", test=True, AUROC_alpha=AUROC_alpha)
            cmss = cmss + [(testcm,)]

        if verbose:
            for computer, logfile in zip(self.models, self.log_files):
                for name, param in computer.named_parameters():
                    logprint(logfile, name)
                    logprint(logfile, param.data.shape)

        for i, (input, target, loss_type, time_length) in enumerate(self.testdl):
            if self.kill_time:
                out_of_time()
            # train
            self.run_one_patient(input, target, loss_type, time_length, cmss, test=True)
            # if i % print_interval == 0:
            #     for model, name, logfile, cms in zip(self.models, self.model_names, self.log_files, cmss):
            #         testcm = cms[0]
            #         cod_loss, toe_loss = testcm.running_loss()
            #         logprint(logfile,
            #                  "%14s " % name + "test batch: %4d. running cod: %.5f, toe: %.5f, total: %.5f" %
            #                  (i, cod_loss, toe_loss, cod_loss + self.beta * toe_loss))
            #         logprint(logfile,
            #                  "%14s " % name + "test sen: %.6f, spe: %.6f, roc: %.6f" % tuple(testcm.running_stats()))

        for model, name, logfile, cms in zip(self.models, self.model_names, self.log_files, cmss):
            testcm = cms[0]
            cod_loss, toe_loss = testcm.running_loss()
            logprint(logfile,
                     "%14s " % name + "test batch: %4d. running cod: %.5f, toe: %.5f, total: %.5f" %
                     (i, cod_loss, toe_loss, cod_loss + self.beta * toe_loss))
            logprint(logfile,
                     "%14s " % name + "test sen: %.6f, spe: %.6f, roc: %.6f" % tuple(testcm.running_stats()))

        self.save_test_stats(cmss, test_stat_dir)

    def run_one_patient(self, input, target, loss_type, time_length, cmss, validate=False, test=False):
        for model, name, optim, logfile, cms, bin, time, is_transformer in \
                zip(self.models, self.model_names, self.optims,
                    self.log_files, cmss, self.binary_criterions, self.time_criterions, self.transformer):
            self.run_one_patient_one_model(model, input, target, loss_type, time_length,
                                           optim, cms, bin, time, validate, test, is_transformer)

    def pre_run_modifier(self, input, target):
        return input, target

    def run_one_patient_one_model(self, computer, input, target, loss_type, time_length,
                                  optim, cms, binary, real, validate=False, test=False, is_transformer=False):
        patient_loss = None
        if not test:
            traincm = cms[0]
            validcm = cms[1]
        else:
            testcm = cms[0]
        # try:
        optim.zero_grad()

        input, target = self.pre_run_modifier(input, target)

        input = Variable(torch.Tensor(input).cuda())
        target = Variable(torch.Tensor(target).cuda())

        # input contains inf values. some lab results. index 4069
        input = input.clamp(-1e10, 1e10)

        loss_type = Variable(loss_type.cuda())

        cause_of_death_target = target[:, 1:]

        if is_transformer:
            # TODO how to generate src_seq?
            total_loss, cod_loss, lat, lem, lvat, toe_loss, patient_output = computer.one_pass(input, time_length,
                                                                                               target, loss_type)
            cause_of_death_output = patient_output[:, 1:]
            sigoutput = torch.sigmoid(cause_of_death_output)
        else:
            patient_output = computer(input)
            cause_of_death_output = patient_output[:, 1:]
            # pdb.set_trace()
            cod_loss = binary(cause_of_death_output, cause_of_death_target)

            toe_output = patient_output[:, 0]
            toe_target = target[:, 0]
            toe_loss = real(toe_output, toe_target, loss_type)

            total_loss = cod_loss + self.beta * toe_loss
            sigoutput = torch.sigmoid(cause_of_death_output)

        if cod_loss.item() < 0:
            import pickle
            with open("debug/itcc.pkl", 'wb') as f:
                pickle.dump((input, target, cause_of_death_output, cause_of_death_target), f)
            print(cod_loss.data[0])
            code.interact(local=locals())
            raise ValueError

        if not validate and not test:
            total_loss.backward()
            optim.step()
            cod_loss = float(cod_loss.item())
            toe_loss = float(toe_loss.item())
            traincm.update_one_pass(sigoutput, cause_of_death_target, cod_loss, toe_loss)
            return cod_loss, toe_loss

        elif validate:
            cod_loss = float(cod_loss.item())
            toe_loss = float(toe_loss.item())
            validcm.update_one_pass(sigoutput, cause_of_death_target, cod_loss, toe_loss)
            return cod_loss, toe_loss
        else:
            assert test
            cod_loss = float(cod_loss.item())
            toe_loss = float(toe_loss.item())
            testcm.update_one_pass(sigoutput, cause_of_death_target, cod_loss, toe_loss)

    def save_test_stats(self, cmss, test_stat_dir):
        for model_name, cms in zip(self.model_names, cmss):
            testcm = cms[0]
            task_dir = os.path.dirname(abspath(__file__))
            if test_stat_dir is None:
                stats_path = Path(task_dir) / "test_stats" / (self.save_str + "_stats")
            else:
                stats_path = Path(task_dir) / test_stat_dir / (self.save_str + "_stats")

            if not os.path.isdir(stats_path):
                os.mkdir(stats_path)

            tp=pd.DataFrame(testcm.true_positive)
            tp.to_csv(stats_path / (model_name + "_tp.csv"))
            tn=pd.DataFrame(testcm.true_negative)
            tn.to_csv(stats_path / (model_name + "_tn.csv"))
            fp=pd.DataFrame(testcm.false_positive)
            fp.to_csv(stats_path / (model_name + "_fp.csv"))
            fn=pd.DataFrame(testcm.false_negative)
            fn.to_csv(stats_path / (model_name + "_fn.csv"))

            conditional=pd.DataFrame({"cp":testcm.conditional_positives, "cn":testcm.conditional_negatives})
            conditional.to_csv(stats_path / (model_name+"_conditional.csv"))
        print("Test stats from cmss saved to", stats_path)


class ExperimentManager(ModelManager):
    def __init__(self, *args, **kwargs):
        super(ExperimentManager, self).__init__(*args, **kwargs)

    def initialize(self, use_cache=True, epoch=None):
        self.init_input_gen(InputGenJ, use_cache=use_cache, collate_fn=pad_collate)
        self.add_DNC()
        self.add_LSTM()
        # self.add_Tacotron()
        if epoch is not None:
            self.load_models(epoch=epoch)
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    def run(self):
        print("Training", self.save_str, self.model_names)
        self.train()

    def test(self, AUROC_alpha=0.0001, verbose=False, test_stat_dir=None):
        """

        :param AUROC_alpha: the step size of AUROC curve
        :param epoch: Select an epoch to test with
        :return:
        """
        print("Testing", self.save_str, self.model_names)
        super(ExperimentManager, self).test(AUROC_alpha=AUROC_alpha, test_stat_dir=test_stat_dir)

    def add_DNC(self):
        param_h = 64  # 64
        param_L = 4  # 4
        param_W = 8  # 8
        param_R = 8  # 8
        param_N = 64  # 64
        param_x = self.param_x
        param_v_t = self.param_v_t

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
        lstm = lstmwrapperG(input_size=self.param_x, output_size=self.param_v_t)
        self.add_model(lstm.cuda(), "lstmG")

    def add_Tacotron(self):
        # add parameters in the hp.file
        taco = Tacotron(self.param_x, self.param_v_t)
        self.add_model(taco.cuda(), "taco")

    def add_APDNC(self):
        param_h = 64  # 64
        param_L = 4  # 4
        param_W = 8  # 8
        param_R = 8  # 8
        param_N = 64  # 64
        param_x = self.param_x
        param_v_t = self.param_v_t

        prior_probability = get_death_code_proportion(self.ig)

        apdnc = APDNC(x=param_x,
                      h=param_h,
                      L=param_L,
                      v_t=param_v_t,
                      W=param_W,
                      R=param_R,
                      N=param_N,
                      prior=prior_probability)

        self.add_model(apdnc.cuda(), "APDNC")

    def overfitting_experiment(self, epoch=None):
        self.save_str = "overfitting"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        # small param DNC
        param_h = 16  # 64
        param_L = 2  # 4
        param_W = 4  # 8
        param_R = 4  # 8
        param_N = 16  # 64
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

        if epoch is not None:
            self.load_models(epoch=epoch)
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    def high_parameters(self, epoch=None):
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

        if epoch is not None:
            self.load_models(epoch=epoch)
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    def high_parameters_1(self, epoch=None):
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

        if epoch is not None:
            self.load_models(epoch=epoch)
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    def high_parameters_2(self, epoch=None):
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

        apdnc = APDNC(x=param_x,
                      h=param_h,
                      L=param_L,
                      v_t=param_v_t,
                      W=param_W,
                      R=param_R,
                      N=param_N,
                      prior=prior_probability)
        self.add_model(apdnc.cuda(), "doubleADNC")

        if epoch is not None:
            self.load_models(epoch=epoch)
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    def adnc_exp(self, epoch=None):
        self.init_input_gen(InputGenJ, collate_fn=pad_collate)
        self.add_APDNC()

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)
        if epoch is not None:
            self.load_models(epoch=epoch)

    def baseline(self, save_str="baseline", epoch=None):
        self.save_str = save_str
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)
        self.add_LSTM()
        self.add_Tacotron()

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def adnc_variations(self, epoch=None):

        self.save_str = "adncvariations"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        param_h = 64
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

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def adnc_variations_1(self, epoch=None):

        self.save_str = "adncvariations1"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        param_h = 64
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

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)



    def adnc_variations_2(self, epoch=None):

        self.save_str = "adncvariations2"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        param_h = 64
        param_L = 4
        param_W = 8
        param_R = 8
        param_N = 64
        param_x = self.param_x
        param_v_t = self.param_v_t
        prior_probability = get_death_code_proportion(self.ig)


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

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def lstm_tacotron_with_prior(self, epoch=None):
        self.save_str = "lowbeta"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        prior_probability = get_death_code_proportion(self.ig)

        lstm = PriorLSTM(input_size=self.param_x, output_size=self.param_v_t, hidden_size=64, num_layers=4,
                         prior=prior_probability).cuda()
        taco = PriorTacotron(self.param_x, self.param_v_t, prior=prior_probability).cuda()

        self.add_model(lstm, "priorlstm")
        self.add_model(taco, "priortaco")

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def lstm_with_prior(self,epoch=None):
        self.save_str = "lstm"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        prior_probability = get_death_code_proportion(self.ig)

        lstm = PriorLSTM(input_size=self.param_x, output_size=self.param_v_t, hidden_size=64, num_layers=4,
                         prior=prior_probability).cuda()

        self.add_model(lstm, "priorlstm")

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)


    def tacotron_with_prior(self, epoch=None):
        self.save_str = "taco"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        prior_probability = get_death_code_proportion(self.ig)

        taco = PriorTacotron(self.param_x, self.param_v_t, prior=prior_probability).cuda()

        self.add_model(taco, "priortaco")

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def dnc_adnc_rerun(self, epoch=None):
        # log was incorrect, and validaion was incorrect. I want to run it again.

        self.save_str = "dnc_adnc_rerun"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        self.add_APDNC()
        self.add_DNC()

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def dnc_rerun(self, epoch=None):

        self.save_str = "dnc_rerun"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        self.add_DNC()

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)


    def adnc_rerun(self, epoch=None):

        self.save_str = "adnc_rerun"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        self.add_APDNC()

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def transformers(self, epoch=None):
        self.save_str = "transformers"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        """
            parser.add_argument('-data', required=True)
        
            parser.add_argument('-epoch', type=int, default=10)
            parser.add_argument('-batch_size', type=int, default=64)
        
            #parser.add_argument('-d_word_vec', type=int, default=512)
            parser.add_argument('-d_model', type=int, default=512)
            parser.add_argument('-d_inner_hid', type=int, default=2048)
            parser.add_argument('-d_k', type=int, default=64)
            parser.add_argument('-d_v', type=int, default=64)
        
            parser.add_argument('-n_head', type=int, default=8)
            parser.add_argument('-n_layers', type=int, default=6)
            parser.add_argument('-n_warmup_steps', type=int, default=4000)
        
            parser.add_argument('-dropout', type=float, default=0.1)
            parser.add_argument('-embs_share_weight', action='store_true')
            parser.add_argument('-proj_share_weight', action='store_true')
        """
        prior_probability = get_death_code_proportion(self.ig)

        binary_criterion = self.binary_criterions()
        real_criterion = self.time_criterions()
        tranforward = TransformerMixedForward(binary_criterion=binary_criterion, real_criterion=real_criterion,
                                              input_size=self.param_x, output_size=self.param_v_t,
                                              prior=prior_probability).cuda()

        binary_criterion = self.binary_criterions()
        real_criterion = self.time_criterions()
        tranattn = TransformerMixedAttn(binary_criterion=binary_criterion, real_criterion=real_criterion,
                                        input_size=self.param_x, output_size=self.param_v_t,
                                        prior=prior_probability).cuda()

        self.add_model(tranforward.cuda(), "tranforward", transformer=True)
        self.add_model(tranattn.cuda(), "tranattn", transformer=True)

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def transformer_with_no_mixed_obj(self, epoch=None):
        self.save_str = "tran_no_mixed_obj"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        """
            parser.add_argument('-data', required=True)

            parser.add_argument('-epoch', type=int, default=10)
            parser.add_argument('-batch_size', type=int, default=64)

            #parser.add_argument('-d_word_vec', type=int, default=512)
            parser.add_argument('-d_model', type=int, default=512)
            parser.add_argument('-d_inner_hid', type=int, default=2048)
            parser.add_argument('-d_k', type=int, default=64)
            parser.add_argument('-d_v', type=int, default=64)

            parser.add_argument('-n_head', type=int, default=8)
            parser.add_argument('-n_layers', type=int, default=6)
            parser.add_argument('-n_warmup_steps', type=int, default=4000)

            parser.add_argument('-dropout', type=float, default=0.1)
            parser.add_argument('-embs_share_weight', action='store_true')
            parser.add_argument('-proj_share_weight', action='store_true')
        """
        prior_probability = get_death_code_proportion(self.ig)

        binary_criterion = self.binary_criterions()
        real_criterion = self.time_criterions()
        tranforward = TransformerMixedForward(binary_criterion=binary_criterion, real_criterion=real_criterion,
                                              input_size=self.param_x, output_size=self.param_v_t,
                                              prior=prior_probability).cuda()

        binary_criterion = self.binary_criterions()
        real_criterion = self.time_criterions()
        tranattn = TransformerMixedAttn(binary_criterion=binary_criterion, real_criterion=real_criterion,
                                        input_size=self.param_x, output_size=self.param_v_t,
                                        prior=prior_probability).cuda()

        self.add_model(tranforward.cuda(), "tranforward", transformer=True)
        self.add_model(tranattn.cuda(), "tranattn", transformer=True)

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def transformer_with_mixed_softmax(self, epoch=None):

        self.save_str = "tran_mixed_softmax"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        prior_probability = get_death_code_proportion(self.ig)

        self.binary_criterions = DiscreteCrossEntropy

        binary_criterion = self.binary_criterions()
        real_criterion = self.time_criterions()
        tranforward = TransformerMixedForwardSoftmax(binary_criterion=binary_criterion, real_criterion=real_criterion,
                                              input_size=self.param_x, output_size=self.param_v_t,
                                              prior=prior_probability, d_model=128, n_layers=4).cuda()

        binary_criterion = self.binary_criterions()
        real_criterion = self.time_criterions()
        tranattn = TransformerMixedAttnSoftmax(binary_criterion=binary_criterion, real_criterion=real_criterion,
                                        input_size=self.param_x, output_size=self.param_v_t,
                                        prior=prior_probability, d_model=128, n_layers=4).cuda()

        self.add_model(tranforward.cuda(), "tranforward", transformer=True)
        self.add_model(tranattn.cuda(), "tranattn", transformer=True)

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def transformer_with_mixed_softmax_forward(self, epoch=None, mixed=True):

        if mixed:
            id="tranmixedforwardsoftmax"
        else:
            id="tranforwardsoftmax"

        self.save_str = id
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        prior_probability = get_death_code_proportion(self.ig)

        self.binary_criterions = DiscreteCrossEntropy

        binary_criterion = self.binary_criterions()
        real_criterion = self.time_criterions()
        tranforward = TransformerMixedForwardSoftmax(binary_criterion=binary_criterion, real_criterion=real_criterion,
                                                     input_size=self.param_x, output_size=self.param_v_t,
                                                     prior=prior_probability, d_model=64,
                                                     d_inner=64, n_layers=4,
                                                     mixed=mixed).cuda()


        self.add_model(tranforward.cuda(), id, transformer=True)

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def transformer_with_mixed_softmax_attn(self, epoch=None, mixed=True):
        if mixed:
            id="tranmixedattnsoftmax"
        else:
            id="tranattnsoftmax"

        self.save_str = id
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        prior_probability = get_death_code_proportion(self.ig)

        self.binary_criterions = DiscreteCrossEntropy


        binary_criterion = self.binary_criterions()
        real_criterion = self.time_criterions()
        tranattn = TransformerMixedAttnSoftmax(binary_criterion=binary_criterion, real_criterion=real_criterion,
                                               input_size=self.param_x, output_size=self.param_v_t,
                                               prior=prior_probability, d_model=64,
                                               d_inner=64, n_layers=4,
                                               mixed=mixed).cuda()

        self.add_model(tranattn.cuda(), id, transformer=True)


        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def softmax_experiment(self, epoch=None):
        # 10 sensitivity is what we have to beat

        self.save_str = "dnc_adnc_softmax"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        self.add_APDNC()
        self.add_DNC()

        self.binary_criterions = DiscreteCrossEntropy
        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def prior_ablation(self,epoch=None):

        self.save_str = "prior_ablation"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        param_h = 64  # 64
        param_L = 4  # 4
        param_W = 8  # 8
        param_R = 8  # 8
        param_N = 64  # 64
        param_x = self.param_x
        param_v_t = self.param_v_t

        computer = PriorDNC(x=param_x,
                            h=param_h,
                            L=param_L,
                            v_t=param_v_t,
                            W=param_W,
                            R=param_R,
                            N=param_N,
                            prior=None)
        self.add_model(computer.cuda(), "DNC")

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)

    def run_simple(self,epoch=None):
        self.save_str = "simple"
        self.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)

        param_x = self.param_x
        param_v_t = self.param_v_t

        computer = Simple(input_size=param_x, target_size=param_v_t)
        self.add_model(computer.cuda(), "simple")

        self.init_optims_and_criterions(torch.optim.Adam, 1e-3)

        if epoch is not None:
            self.load_models(epoch=epoch)




def main():
    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
    ds.lstm_with_prior(epoch=0)
    ds.run()
    # ds.test(AUROC_alpha=0.001)

    # ds.adnc_variations()

def main2():
    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=20)
    ds.tacotron_with_prior(epoch=0)
    ds.run()
    # ds.test(AUROC_alpha=0.001)

    # ds.adnc_variations()

def main3():


    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=20)
    ds.adnc_variations_1(epoch=None)
    ds.run()


    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=20)
    ds.transformer_with_mixed_softmax_forward(epoch=None, mixed=False)
    ds.run()

def main4():

    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=20)
    ds.adnc_variations_2(epoch=None)
    ds.run()

def main5():
    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=20)
    ds.softmax_experiment(epoch=None)
    ds.run()

def main6(epoch=None):
    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=20)
    ds.save_str = "tranattnsoftmax"
    ds.init_input_gen(InputGenJ, collate_fn=pad_collate, use_cache=True)
    prior_probability = get_death_code_proportion(ds.ig)

    ds.binary_criterions = DiscreteCrossEntropy

    binary_criterion = ds.binary_criterions()
    real_criterion = ds.time_criterions()
    tranattn = TransformerMixedAttn(binary_criterion=binary_criterion, real_criterion=real_criterion,
                                    input_size=ds.param_x, output_size=ds.param_v_t,
                                    prior=prior_probability).cuda()

    ds.add_model(tranattn.cuda(), "tranattnsoftmax", transformer=True)

    ds.init_optims_and_criterions(torch.optim.Adam, 1e-3)

    if epoch is not None:
        ds.load_models(epoch=epoch)
    ds.run()

def main7():
    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=30)
    ds.prior_ablation(epoch=0)
    ds.run()

# alphabetical order, as plots
# run another 20 epochs
def all_working_1():
    def e1():
        ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=41)
        ds.dnc_rerun(epoch=0)
        ds.run()
        del ds

    def e2():
        ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=41)
        ds.adnc_variations_1(epoch=0)
        ds.run()
        del ds

    def e3():
        ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=41)
        ds.prior_ablation(epoch=0)
        ds.run()
        del ds

    def e4():
        ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=41)
        ds.run_simple(epoch=0)
        ds.run()
        del ds

    def e5():
        ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=41)
        ds.lstm_with_prior(epoch=0)
        ds.run()

    # e1()
    # e2()
    # e3()
    # e4()
    e5()

def all_working_2():
    with torch.cuda.device(1):
        def e1():
            ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=41)
            ds.tacotron_with_prior(epoch=0)
            ds.run()
            del ds

        def e2():
            ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=41)
            ds.adnc_rerun(epoch=0)
            ds.run()
            del ds

        def e3():
            ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=41)
            ds.softmax_experiment(epoch=0)
            ds.run()
            del ds

        def e4():
            ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=41)
            ds.adnc_variations_2(epoch=0)
            ds.run()
            del ds

        e1()
        e2()
        e3()
        e4()

def test_all_for_AUROC_0(epoch=0, AUROC_alpha=0.0001, test_stat_dir=None):
    # incredibly messy https://github.com/pytorch/pytorch/issues/8637
    # the model has to load on the correct gpu, otherwise it does not work at all.
    # I suggest that you look into the save_models function to not save methods

    # you won't be able to run all tests on V100. memory does not get freed comlpetely.


    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
    ds.lstm_with_prior(epoch=epoch)
    ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)

    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
    ds.dnc_rerun(epoch=epoch)
    ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)

    # transformers
    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=20)
    ds.transformer_with_mixed_softmax_forward(epoch=epoch, mixed=False)
    ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)

    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=20)
    ds.transformer_with_mixed_softmax_forward(epoch=epoch, mixed=True)
    ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)
    del ds

    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=20)
    ds.transformer_with_mixed_softmax_attn(epoch=epoch, mixed=False)
    ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)
    del ds

    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
    ds.adnc_variations_1(epoch=epoch)
    ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)
    del ds

    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
    ds.prior_ablation(epoch=epoch)
    ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)
    del ds

    # ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
    # ds.run_simple(epoch=epoch)
    # ds.test(AUROC_alpha=AUROC_alpha, test_stat_dir=test_stat_dir)
    # del ds


def test_all_for_AUROC_1(epoch=0, AUROC_alpha=0.0001, test_stat_dir=None):
    def e1():
        with torch.cuda.device(1):
            ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
            ds.tacotron_with_prior(epoch=epoch)
            ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)

    def e2():
        with torch.cuda.device(1):
            ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
            ds.adnc_rerun(epoch=epoch)
            ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)

    def e5():
        with torch.cuda.device(1):
            ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=20)
            ds.transformer_with_mixed_softmax_attn(epoch=epoch, mixed=True)
            ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)
            del ds

    def e3():
        with torch.cuda.device(1):
            ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
            ds.softmax_experiment(epoch=epoch)
            ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)
            del ds
    def e4():
        with torch.cuda.device(1):
            ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
            ds.adnc_variations_2(epoch=epoch)
            ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)
            del ds

    # e1()
    # e2()
    # e5()
    e3()
    e4()

def transformer200():
    with torch.cuda.device(1):
        ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=21)
        ds.transformer_with_mixed_softmax_attn(epoch=0, mixed=True)
        ds.run()

def transformer201():
    with torch.cuda.device(0):
        # transformers
        ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=21)
        ds.transformer_with_mixed_softmax_forward(epoch=0, mixed=False)
        ds.run()

        ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=21)
        ds.transformer_with_mixed_softmax_forward(epoch=0, mixed=True)
        ds.run()

        ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=21)
        ds.transformer_with_mixed_softmax_attn(epoch=0, mixed=False)
        ds.run()


def retest_lstm(epoch=4, AUROC_alpha=0.0001, test_stat_dir="earlystopping_stats"):
    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
    ds.lstm_with_prior(epoch=epoch)
    ds.test(AUROC_alpha=AUROC_alpha,test_stat_dir=test_stat_dir)

if __name__ == '__main__':
    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=40)
    ds.run_simple(epoch=0)
    ds.test(AUROC_alpha=0.0001)
    del ds
