from death.post.inputgen_planJ import InputGenJ
import torch.nn as nn
import torch
import pandas as pd
from collections import Counter
import tqdm
import numpy as np

class Dumb(nn.Module):
    def __init__(self):
        super(Dumb, self).__init__()

        ig = InputGenJ(no_underlying=True, death_only=True)
        series = ig.get_series("death", "code")
        index_list = []
        dic = ig.death_code_dict
        total_patients = len(ig.death_rep_person_id)  # equals to the intersection
        patients_lookup = {id: 0 for id in ig.death_rep_person_id}

        for index, row in tqdm.tqdm(ig.death.iterrows()):
            if index[0] in patients_lookup:
                code = row['code']
                idx = dic[code]
                index_list.append(idx)

        # for row in tqdm.tqdm(series):
        #     idx=dic[row]
        #     index_list.append(idx)

        counter = Counter(index_list)
        code_proportion = list(counter.values())

        for i in range(len(code_proportion)):
            code_proportion[i] /= total_patients

        self.pred = np.array([0]+code_proportion)

    def forward(self, input):
        # it predicts background distribution, and for time it predicts zero.
        # let's see how it compares.

        # no parameters, no backward
        # this is the true baseline. You gotta do better than this right?
        return self.pred

