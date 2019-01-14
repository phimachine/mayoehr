from death.DNC.nobntrainer import main
import torch
from death.final.killtime import *
import multiprocessing
import time
if __name__=='__main__':
    with torch.cuda.device(0):
        main(load=False,lr=1e-3,savestr="nobn",beta=1e-5)
