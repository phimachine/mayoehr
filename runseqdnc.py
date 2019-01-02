from death.DNC.seqtrainer import main
import torch
from death.final.killtime import *
import multiprocessing
import time
if __name__=='__main__':
    with torch.cuda.device(1):
        main(load=True,lr=0.0003,savestr="poswei",beta=1e-5)
