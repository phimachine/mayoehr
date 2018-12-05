from death.DNC.batchtrainer import main
import torch

with torch.cuda.device(1):
    main(load=False,lr=1e-3, savestr="small",curri=False)
