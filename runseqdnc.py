from death.DNC.seqtrainer import main
import torch

with torch.cuda.device(1):

    main(load=True,lr=1e-3, savestr="retrain",curri=False)
