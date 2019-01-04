from death.DNC.seqtrainer import main
import torch
from death.final.killtime import *
import multiprocessing
import time
if __name__=='__main__':
    with torch.cuda.device(0):
        main(load=False,lr=0.001,savestr="poswei",beta=1e-5)
