# to do hyperparameter tuning, we need to establish a way to compare models
# 1, After ~5000 training iterations, we start to evaluate the performance
# 2, We keep a moving average of validation errors to smooth out the val loss
# 3, Once the moving average no longer decreases (for a few times), we stop early and record the performance.
#    Moving average no longer decreases if later performance is worse than much earlier performances "consistently".

# Why can't we use this scheme to tune all neural networks. The dimensionality guarantees that the parameters will
# always be optimized? If not, then why is this hyperparameter search design a good idea?

from collections import deque, OrderedDict
from death.post.inputgen_planG import InputGenG, pad_collate
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch
import datetime
from death.DNC.seqDNC import SeqDNC
from death.baseline.lstmtrainerG import lstmwrapperG
import pickle

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def get_log_file():
    timestring = str(datetime.datetime.now().time())
    timestring.replace(" ", "_")
    return "log/hyper_" + timestring

def logprint(logfile, string, log_only=False):
    if logprint is not None and logprint !=False:
        with open(logfile, 'a') as handle:
            handle.write(string)
    if not log_only:
        print(string)

class HyperParameterTuner():

    """
    Generic hyper parameter tuner with 2 based parameter space.
    Search for pareto equilibrium.
    Early stopping validation given 5 chances to increase.
    """

    def __init__(self, mode="DNC",load=False):
        # this is the set of the best parameters I have so far, or the one I'm testing
        self.parameters = {}
        self.param_list=[]

        # a copy of the parameters when it worked, in case exception handling needs it
        self.best_parameters={}

        self.total_tuning_rounds = 100
        self.minimum_training_iterations = 2000

        self.best_parameters={}
        # needs to set up to be some value
        self.best_validation=0.0024667

        self.mode = mode

        self.num_workers=16

        self.last_tuned=None

        self.debug=True

        if self.mode == "DNC":
            ig = InputGenG(death_fold=0)
            trainds = ig.get_train()
            validds = ig.get_valid()
            testds = ig.get_test()
            self.bs = 8
            self.traindl = DataLoader(dataset=trainds, batch_size=self.bs, num_workers=self.num_workers, collate_fn=pad_collate)
            self.traindl = iter(self.traindl)
            self.validdl = DataLoader(dataset=validds, batch_size=self.bs, num_workers=self.num_workers, collate_fn=pad_collate)
            self.validdl = iter(self.validdl)
            self.binary_criterion= nn.BCEWithLogitsLoss()
            self.optimizer_type=torch.optim.Adam
            self.lr=1e-3
            self.parameters=OrderedDict()
            self.parameters['h']=64
            self.parameters['L']=4
            self.parameters['W']=8
            self.parameters['R']=8
            self.parameters['N']=64
            self.best_parameters=self.parameters.copy()
            self.param_list=list(self.parameters)

        elif self.mode == "LSTM":
            ig = InputGenG(death_fold=0)
            trainds = ig.get_train()
            validds = ig.get_valid()
            testds = ig.get_test()
            self.bs=8
            self.traindl = DataLoader(dataset=trainds, batch_size=self.bs, num_workers=self.num_workers, collate_fn=pad_collate)
            self.traindl = iter(self.traindl)
            self.validdl = DataLoader(dataset=validds, batch_size=self.bs, num_workers=self.num_workers, collate_fn=pad_collate)
            self.validdl = iter(self.validdl)
            self.binary_criterion= nn.BCEWithLogitsLoss()
            self.optimizer_type=torch.optim.Adam
            self.bs=256
            self.lr=1e-2
            self.parameters=OrderedDict()
            self.parameters['h']=128
            self.parameters['L']=32

            self.best_parameters=self.parameters.copy()
            self.param_list=list(self.parameters)

        elif self.mode == "Tacotron":
            pass
        else:
            raise ValueError("Not supported")

        # choose an efficient validation size here
        self.valid_records=256
        self.average_step_in_records=64
        # how many validations will be ignored before tuning begins
        self.ignore_valid=5

        timestring=str(datetime.datetime.now().time())
        timestring.replace(" ","_")
        self.param_log=get_log_file()

        if load:
            self.load_best_param()

    def save_best_param(self):
        with open("best_parameters.pkl",'wb') as f:
            pickle.dump((self.parameters,self.best_validation, self.bs, self.last_tuned), f)

    def load_best_param(self):
        with open('best_parameters.pkl','rb') as f:
            self.parameters,self.best_validation, self.bs, self.last_tuned=pickle.load(f)


    def tune(self):
        # main function to be called
        # tuning each parameter one by one
        # the parameter is chosen greedily. If changing a parameter yields good results, then the same change is applied
        # again, until it does not improve, and then next parameter is chosen.

        logprint(self.param_log, "Starting parameters:")
        logprint(self.param_log, str(self.parameters))
        logprint(self.param_log, "workers: "+str(self.num_workers)+", batch size:"+str(self.bs))

        def greedily_tune(bigger, tune_streak, parameter):
            changed=True
            value=self.parameters[parameter]
            while changed:
                logprint(self.param_log, "Tuning " + str(parameter) + ", original: " + str(value)+" bigger: ", str(bigger))
                if bigger:
                    value = value * 2
                else:
                    if value > 1:
                        value = value // 2
                    else:
                        '''The parameter cannot be modified further'''
                        break
                self.parameters[parameter]=value
                # The function below will modify self.parameters or self.best_parameters

                changed = self.tune_one_param(bigger)
                if changed:
                    tune_streak += 1

            return tune_streak

        stable=False
        while not stable:
            stable=True

            starting_param_index=self.last_tuned
            if starting_param_index is None:
                starting_param_index=0

            param_index=starting_param_index
            while True:
                # go through all parameters starting with the one specified
                parameter=self.param_list[param_index]
                tune_streak=0

                # Try smaller first, because we are overfitting.
                # Note that if smaller does not work, bigger is always tested.
                if np.random.random()>0.3:
                    bigger=True
                else:
                    bigger=False

                tune_streak=greedily_tune(bigger, tune_streak, parameter)

                if tune_streak==0:
                    # if the guessed direction did not lead to a better outcome,
                    # try the other direction
                    bigger = not bigger
                    tune_streak=greedily_tune(bigger, tune_streak, parameter)

                if tune_streak!=0:
                    logprint(self.param_log,
                             "Tuned parameter"+str(parameter)+"to be"+str(self.parameters[parameter])+"with"+str(tune_streak)+str("tries"))
                    stable=False

                if param_index==len(self.param_list)-1:
                    param_index=0
                else:
                    param_index+=1
                self.last_tuned=param_index
                if param_index==starting_param_index:
                    break


            # at this point, stable will only be False if every parameter has been probed in both directions and
            # we have no improvement. This is the definition of pareto equilibrium.


    def tune_one_param(self,bigger):
        if self.mode=="DNC":
            from death.post.channelmanager import ChannelManager
        else:
            from death.baseline.lstmcm import ChannelManager

        if bigger:
            self.bs //= 2
        else:
            self.bs *= 2

        try:
            # initialize with the current parameters
            self.init()

            # self.traincm = ChannelManager(self.traindl, self.bs, model=self.model)
            # self.validcm = ChannelManager(self.validdl, self.bs, model=self.model)
            self.optimizer = self.optimizer_type([i for i in self.model.parameters() if i.requires_grad], lr=self.lr)

            # TODO I expect this function to be wrapped in a try catch clause, but I'm not sure which
            # exception to catch yet for insufficient memory.
            best_validation=self.early_stopping()
        except RuntimeError:
            self.bs/=2
            self.init()

            # self.traincm = ChannelManager(self.traindl, self.bs, model=self.model)
            # self.validcm = ChannelManager(self.validdl, self.bs, model=self.model)
            self.optimizer = self.optimizer_type([i for i in self.model.parameters() if i.requires_grad], lr=self.lr)

            best_validation=self.early_stopping()

        # this is necessary to terminate workers so we don't run into spawn limit.
        del self.traindl, self.validdl

        if best_validation<self.best_validation:
            self.best_parameters=self.parameters.copy()
            self.save_best_param()
            return True
        else:
            self.parameters=self.best_parameters.copy()
            # if not improved, batch size needs to be reverted too
            if bigger:
                self.bs*=2
            else:
                self.bs/=2
            return False


    def early_stopping(self):
        """
        Use early stopping criterion to evaluate the parameter chosen.
        :return: evaluation with validation of best performance before consistent increase
        """

        best_validation=None
        stopping=5

        # The first 5 validations will be thrown away
        for _ in range(self.ignore_valid):
            train_loss, validation_loss= self.run()
            best_validation=validation_loss

        counter=0
        # the early stopping criterion is very simple
        # if the current validation does not beat the best validation loss in 5 consecutive runs, then record the
        # second best_validation_loss and call it off

        while counter!=stopping:
            train_loss, validation_loss= self.run()
            if validation_loss<best_validation:
                best_validation=validation_loss
                counter=0
                print("Validation down, best validation: "+str(best_validation))
            else:
                print("Validation up, best validation: "+str(best_validation))
                counter+=1

        return best_validation

    def run(self):
        """
        Train the model for a few rounds and get one validation
        :param model:
        :return:
        """

        # Validation size is measured by the mount of validation data points.
        # Training size is measured in computation cycles.
        # This design discourages using huge parameters and very small batch size, which will certainly dominate
        # with limited computation resources.

        # train cycle: how many times run_one_step_DNC will be called for every validation score
        # e.g. 1000 training cycles with 128 batch size will lead to 128,000 time steps per validation access

        print_interval = 100
        train_cycle = 128000 // self.bs
        training_losses = []
        for cycle in range(train_cycle):
            train_step_loss = self.run_one_seq().data[0]
            training_losses.append(train_step_loss)
            if cycle % print_interval == 0:
                logprint(self.param_log,
                         "Internal cycle " + str(cycle) + "/" + str(train_cycle) + " with loss " + str(train_step_loss))
        # this is only for reference purposes
        average_training_loss = np.mean(training_losses)
        validation = self.validate()
        logprint(self.param_log, "Validation: " + str(validation))

        return average_training_loss, validation


    def validate(self):
        val_batches=self.valid_records*self.average_step_in_records//self.bs
        training_losses=[]
        for batch in range(val_batches):
            loss=self.valid_one_step().data[0]
            training_losses.append(loss)
        return np.mean(training_losses)


    def run_one_seq(self):
        self.model.train()
        self.optimizer.zero_grad()
        input, target, loss_type = next(self.traindl)
        input = Variable(input).cuda()
        target = Variable(target).cuda()
        loss_type = Variable(loss_type).cuda()

        try:

            output = self.model(input)

            time_to_event_output = output[:, 0]
            cause_of_death_output = output[:, 1:]
            time_to_event_target = target[:, 0]
            cause_of_death_target = target[:, 1:]

            loss = self.binary_criterion(cause_of_death_output, cause_of_death_target)
            loss.backward()
            self.optimizer.step()
            if self.debug:
                assert (loss.data[0] == loss.data[0])
            return loss
        except AssertionError:
            for key, val in self.model.lstm._parameters.items():
                if (val != val).any():
                    print(key, val)
            raise AssertionError


    def valid_one_step(self):
        self.model.eval()
        input, target, loss_type, states_tuple = next(self.validcm)
        target = target.squeeze(1)
        input = Variable(input).cuda()
        target = Variable(target).cuda()
        loss_type = Variable(loss_type).cuda()
        self.model.assign_states_tuple(states_tuple)

        try:
            if self.debug:
                for state in states_tuple:
                    assert (state == state).all()


                output, states_tuple = self.model(input)
                self.validcm.push_states(states_tuple)
                if self.debug:
                    for state in states_tuple:
                        assert (state==state).all()

                    assert (output==output).all()

                time_to_event_output = output[:, 0]
                cause_of_death_output = output[:, 1:]
                time_to_event_target = target[:, 0]
                cause_of_death_target = target[:, 1:]

                loss = self.binary_criterion(cause_of_death_output, cause_of_death_target)
                if self.debug:
                    assert (loss.data[0]==loss.data[0])
                return loss
        except AssertionError:
            for key, val in self.model.lstm._parameters.items():
                if (val != val).any():
                    print(key, val)
            raise AssertionError

    # def init_DNC(self):
    #     self.model=DNC(x=69505,v_t=5952,bs=self.bs, **self.parameters)
    #     self.model=self.model.cuda()

    def init(self):
        if self.mode=="DNC":
            self.model = SeqDNC(x=66529, v_t=5952, bs=self.bs, **self.parameters)
        else:
            self.model=lstmwrapperG(hidden_size=self.parameters["h"], num_layers=self.parameters["L"])
        self.model=self.model.cuda()


def main(mode="DNC"):
    ht = HyperParameterTuner(mode=mode)
    ht.tune()

if __name__=="__main__":
    main(mode="LSTM")