from death.taco.tacotrainer import main
import torch

with torch.cuda.device(0):
    main(load=False, savestr="taco")
