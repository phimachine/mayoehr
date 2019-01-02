from death.baseline.lstmtrainerG import *
import torch

with torch.cuda.device(1):
    main(load=True,savestr="5toe", lr=1e-3,beta=1e-5)