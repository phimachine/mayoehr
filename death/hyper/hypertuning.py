# to do hyperparameter tuning, we need to establish a way to compare models
# 1, After ~5000 training iterations, we start to evaluate the performance
# 2, We keep a moving average of validation errors to smooth out the val loss
# 3, Once the moving average no longer decreases (for a few times), we stop early and record the performance.
#    Moving average no longer decreases if later performance is worse than much earlier performances "consistently".

# Why can't we use this scheme to tune all neural networks. The dimensionality guarantees that the parameters will
# always be optimized? If not, then why is this hyperparameter search design a good idea?

from collections import deque, OrderedDict
from death.post.inputgen_planE import InputGenE
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch

class HyperParameterTuner():

    def __init__(self, mode="BatchDNC"):
        # this is the set of the best parameters I have so far, or the one I'm testing
        self.parameters = {}
        # a copy of the parameters when it worked, in case exception handling needs it
        self.best_parameters={}

        self.total_tuning_rounds = 100
        self.minimum_training_iterations = 2000

        self.best_parameters={}
        self.best_validation=None

        self.mode = mode
        if self.mode == "BatchDNC":
            self.ig=InputGenE()
            trainds=self.ig.get_train_dataset()
            validds=self.ig.get_valid_dataset()
            self.traindl = DataLoader(dataset=trainds, batch_size=1, num_workers=self.num_workers)
            self.validdl = DataLoader(dataset=validds, batch_size=1, num_workers=self.num_workers)
            self.binary_criterion= nn.BCEWithLogitsLoss()
            self.optimizer_type=torch.optim.Adam
            self.lr=1e-3
        else:
            raise ValueError("Not supported")


        self.num_workers=8
        self.bs=32

        # choose an efficient validation size here
        self.valid_records=256
        self.average_step_in_records=64
        # how many validations will be ignored before tuning begins
        self.ignore_valid=5


    def tune(self):
        # main function to be called
        # tuning each parameter one by one
        # the parameter is chosen greedily. If changing a parameter yields good results, then the same change is applied
        # again, until it does not improve, and then next parameter is chosen.



    def tune_one_param(self, parameter, bigger):
        """
        Tune one parameter
        :param parameter: the parameter to be tuned
        :param bigger: direction of change
        :return:
        Whether this change betters the result
        The result
        """

        if self.mode=="BatchDNC":
            return self.tune_one_param_DNC(parameter,bigger)

    def tune_one_param_DNC(self, parameter, bigger):
        """
        Tunes one parameter that is given

        Compare the performance of this parameter set with the best parameter set we have
        Iff better, replace
        :param parameter:
        :param bigger:
        :return: If the parameter if better
        """

        from death.post.channelmanager import ChannelManager
        # bs is a parameter dependent upon the parameter set, because of memory constraint
        self.traincm = ChannelManager(self.traindl, self.getbs(), model=self.model)
        self.validcm = ChannelManager(self.validdl, self.getbs(), model=self.model)

        # modify the parameter
        if bigger:
            self.parameters[parameter]*=2
        else:
            if self.parameters!=1:
                self.parameters[parameter]/=2
            else:
                return False

        # initialize with the current parameters
        self.model = self.init_DNC()
        self.optimizer = self.optimizer_type([i for i in self.model.parameters() if i.requires_grad], lr=self.lr)

        self.run=self.run_DNC
        best_validation=self.early_stopping()
        if best_validation>self.best_validation:
            self.best_parameters=self.parameters
            return True
        else:
            self.parameters=self.best_parameters
            return False

    def early_stopping(self):

        best_validation=None

        # The first 5 validations will be thrown away
        for _ in range(self.ignore_valid):
            train_loss, validation_loss= self.run()
            best_validation=validation_loss

        counter=0
        # the early stopping criterion is very simple
        # if the current validation does not beat the best validation loss in 5 consecutive runs, then record the
        # second best_validation_loss and call it off

        while counter!=5:
            train_loss, validation_loss= self.run()
            if validation_loss<best_validation:
                best_validation=validation_loss
                counter=0
                print("Validation decreasing")
            else:
                print("Validation increasing")
                counter+=1

        return best_validation

    def getbs(self):
        pass


    def run_DNC(self):
        """
        Train the model for a few rounds and get one validation
        :param model:
        :return:
        """


        # Validation size is determined by the mount of validation data points.
        # Training size is measured in computation cycles.
        # This design discourages using huge parameters and very small batch size, which will certainly dominate
        # with limited computation resources.

        # train cycle: how many times run_one_step_DNC will be called for every validation score
        train_cycle=1000
        training_losses=[]
        for cycle in range(train_cycle):
            train_step_loss=self.run_one_step_DNC()[0]
            training_losses.append(train_step_loss)
        # this is only for reference purposes
        average_training_loss=np.mean(training_losses)
        validation=self.validate_DNC()

        return average_training_loss, validation


    def run_one_step_DNC(self):
        self.model.train()
        self.optimizer.zero_grad()
        input, target, loss_type, states_tuple = next(self.traincm)
        target = target.squeeze(1)
        input = Variable(input).cuda()
        target = Variable(target).cuda()
        loss_type = Variable(loss_type).cuda()
        self.model.assign_states_tuple(states_tuple)
        output, states_tuple = self.model(input)
        self.traincm.push_states(states_tuple)

        time_to_event_output = output[:, 0]
        cause_of_death_output = output[:, 1:]
        time_to_event_target = target[:, 0]
        cause_of_death_target = target[:, 1:]

        loss = self.binary_criterion(cause_of_death_output, cause_of_death_target)
        loss.backward()
        self.optimizer.step()
        return loss


    def valid_one_step_DNC(self):
        self.model.eval()
        input, target, loss_type, states_tuple = next(self.traincm)
        target = target.squeeze(1)
        input = Variable(input).cuda()
        target = Variable(target).cuda()
        loss_type = Variable(loss_type).cuda()
        self.model.assign_states_tuple(states_tuple)
        output, states_tuple = self.model(input)
        self.traincm.push_states(states_tuple)

        time_to_event_output = output[:, 0]
        cause_of_death_output = output[:, 1:]
        time_to_event_target = target[:, 0]
        cause_of_death_target = target[:, 1:]

        loss = self.binary_criterion(cause_of_death_output, cause_of_death_target)
        return loss


    def init_DNC(self):
        model=None
        return model


    def init_parameters(self):
        if self.mode=="BatchDNC":
            return self.init_parameters_DNC()

    def init_parameters_DNC(self):
        self.parameters = OrderedDict([("x", 69505),
                                       ("h", 128),
                                       ("L", 4),
                                       ("v_t", 5952),
                                       ("W", 8),
                                       ("R", 4),
                                       ("N", 64)])

    def validate_DNC(self):
        val_batches=self.valid_records*self.average_step_in_records/self.bs
        training_losses=[]
        for batch in range(val_batches):
            training_losses.append(self.valid_one_step_DNC())
        return np.mean(training_losses)

