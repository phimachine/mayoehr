# to do hyperparameter tuning, we need to establish a way to compare models
# 1, After ~5000 iterations, we start to evaluate the performance
# 2, We keep a moving average of validation errors to smooth out the val loss
# 3, Once the moving average no longer decreases (for a few times), we stop early and record the performance.
#    Moving average no longer decreases if later performance is worse than much earlier performances "consistently".

from collections import deque

class HyperParameterTuner():

    def __init__(self):
        # initialize the parameters
        self.parameters={}
        self.rldmax_len=10

    def tune(self):
        # tuning each parameter one by one
        # the parameter is chosen greedily. If changing a parameter yields good results, then the same change is applied
        # again, until it does not improve, and then next parameter is chosen.

        running_loss_deque = deque(maxlen=self.rldmax_len)
        running_loss_deque.appendleft(self.report())


    def report(self):
        # this function runs the model and validate once
        # you should validate with batch size and frequency set appropriately here.
        validation_error=None
        raise NotImplementedError
        return validation_error