from death.baseline.lstmtrainerG import *
import torch

with torch.cuda.device(1):
    main(load=True,savestr="lowlr", lr=1e-2)