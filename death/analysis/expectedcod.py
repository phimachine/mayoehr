from death.post.inputgen_planJ import InputGenJ
import pandas as pd
from collections import Counter
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from death.post.inputgen_planH import InputGenH, pad_collate
from death.analysis.expectedroc import get_death_code_proportion

def calcexpcod():
    ig=InputGenJ()
    bs=32
    tr=ig.get_train_cached()
    bce=nn.BCELoss()
    losses = []
    prior_probability = get_death_code_proportion(ig)
    output=torch.from_numpy(prior_probability).float().cuda()
    output=output.unsqueeze(0).repeat(bs, 1)
    tr= DataLoader(dataset=tr, batch_size=bs, num_workers=8,collate_fn=pad_collate)

    for idx,data in tqdm.tqdm(enumerate(tr)):
        input, target, loss_type, time_length=data
        cause_of_death_target = target[:,1:].cuda()
        cause_of_death_target=cause_of_death_target.float()
        try:
            cod_loss=bce(output,cause_of_death_target)
        except ValueError:
            cod_loss=bce(output[:cause_of_death_target.shape[0],],cause_of_death_target)
        losses.append(cod_loss.item())

    # 0.09312667474150657
    print(sum(losses)/len(losses))
