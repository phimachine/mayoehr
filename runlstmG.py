from death.baseline.lstmtrainerG import *
import torch

with torch.cuda.device(1):
    main(load=True,savestr="maxpool", lr=1e-5,beta=1e-5)