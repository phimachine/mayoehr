from death.taco.tacotrainer import *
import torch

with torch.cuda.device(0):
    main(load=True, savestr="taco")
