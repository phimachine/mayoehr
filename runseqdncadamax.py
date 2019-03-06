from death.DNC.adamaxseqtrainer import main
import torch
from death.final.killtime import *
import multiprocessing
import time
if __name__=='__main__':
    with torch.cuda.device(0):
        main(load=True,lr=1e-3,savestr="newtarget",beta=1e-5, kill_time=False)

# the sensitivity is still 10%, after eliminating the rare codes.
# this is reasonable. The rare codes appear not often enough to have a big impact on the sensitivity.
# what's the average code frequency?
# for a code that appears one in a million, a sensitivity of 10% is quite good. No?
# what is the expected sensitivity for a sparse code if the classifier categorizes randomly? Gini Impurity right?
# it seems training with all death records does not actually increase the sensitivity too much?
# Try positive weights again.
# Do hyperparameter tuning again.
