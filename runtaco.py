from death.taco.tacotrainer import *
import torch

with torch.cuda.device(1):
    main(load=True, savestr="poswei")
