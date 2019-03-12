from death.DNC.priortrainer import main
import torch
from death.final.killtime import *
import multiprocessing
import time
if __name__=='__main__':
        main(load=False,lr=1e-3,savestr="prior",beta=1e-5, kill_time=False)

# stuck at 0.69315