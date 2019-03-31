from death.baseline.lstmtrainerG import *
import torch

with torch.cuda.device(0):
    main(load=True,savestr="noposwei", lr=1e-3,beta=1e-5)