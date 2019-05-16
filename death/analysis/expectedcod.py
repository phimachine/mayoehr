from death.post.inputgen_planJ import InputGenJ
import pandas as pd
from collections import Counter
import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from death.post.inputgen_planH import InputGenH, pad_collate

def get_death_code_proportion(ig):
    index_list=[]
    dic=ig.death_code_dict
    total_patients=len(ig.death_rep_person_id) # equals to the intersection
    patients_lookup={id: 0 for id in ig.death_rep_person_id}

    patient_list = []
    last_patient = None

    for index, row in tqdm.tqdm(ig.death.iterrows(), total=ig.death.shape[0]):
        rep_id=index[0]
        if rep_id in patients_lookup:
            if rep_id != last_patient:
                # new patient
                index_list+=patient_list
                patient_list = []
                last_patient=rep_id

            code = row['code']
            while code is not "":
                idx=dic[code]
                patient_list.append(idx)
                code=code[:-1]
            patient_list=list(set(patient_list))
    index_list += patient_list

    # for row in tqdm.tqdm(series):
    #     idx=dic[row]
    #     index_list.append(idx)
    #
    counter=Counter(index_list)
    # code_proportion=list(counter.values())
    #
    # for i in range(len(code_proportion)):
    #     code_proportion[i]/=total_patients

    prop=np.zeros((ig.output_dim-1))

    for key, value in counter.items():
        prop[key]+=value
        if value==0 or value<0:
            print("look into it.")
    prop=prop/total_patients

    return prop

# these are the debugging functions after seeing ROC is bigger than 1 for trivial predictions
# average of ROC, not ROC of average

def ROC_3():
    """
    Well, the ROC_2() apparently does not take into consideration of the strcutural codes.
    These things are tricky as hell. Unexpected.
    Well why did you not use NP?
    Why was ROC_2() formula still wrong? Jesus.
    :return:
    """

    ig=InputGenJ(no_underlying=True, death_only=True)

    print("Generating prior")
    code_proportion=get_death_code_proportion(ig)
    code_proportion=np.asarray(code_proportion)

    # actual positives is code proportion
    expected_true_positives=(code_proportion*code_proportion)

    setp=sum(expected_true_positives)
    scp=sum(code_proportion)
    sensitivity_average=expected_true_positives/code_proportion


    print("Expected true positives per patient", setp) # 1.276
    print("Expected positives per patient", scp) # 10.7094
    print("Expected sensitivity:",np.mean(sensitivity_average)) # 0.02467
    # higher than actual

    # what's the expected specificity?
    negative_code_proportion=1-code_proportion
    expected_true_negatives=negative_code_proportion*negative_code_proportion
    specificity_average=expected_true_negatives/negative_code_proportion
    actual_negatives=sum(negative_code_proportion)
    true_negatives=sum(expected_true_negatives)

    # what is the Gini impurity compared to the AUC?
    print("Expected true negatives per patient", true_negatives) # 413.85
    print("Expected negatives per patient", actual_negatives) # 423.29
    print("Expected specificity:",np.mean(specificity_average)) # 0.97532

    # what is the expected binary cross entropy with random guess?
    expected_BCE=-code_proportion*np.log(code_proportion)-negative_code_proportion*np.log(negative_code_proportion)

    print(np.mean(expected_BCE)) # 0.09295, which is close to the loss on the first validation before training

def ROC_2():
    """
    Because codes have been merged, and we do not know the interaction among the merged codes,
    I have to get the ROC from the data itself.
    It's very fast to generate numpy with the dataloader, so I have to count the pandas
    :return:
    """

    ig=InputGenJ(no_underlying=True, death_only=True)
    series=ig.get_series("death","code")
    index_list=[]
    dic=ig.death_code_dict
    total_patients=len(ig.death_rep_person_id) # equals to the intersection
    patients_lookup={id: 0 for id in ig.death_rep_person_id}

    for index, row in tqdm.tqdm(ig.death.iterrows()):
        if index[0] in patients_lookup:
            code=row['code']
            idx=dic[code]
            index_list.append(idx)

    # for row in tqdm.tqdm(series):
    #     idx=dic[row]
    #     index_list.append(idx)

    counter=Counter(index_list)
    code_proportion=list(counter.values())

    for i in range(len(code_proportion)):
        code_proportion[i]/=total_patients

    expected_true_positives=[]
    for prop in code_proportion:
        # a code has 30% chance happening
        # 30% background guess
        # true positive exepctation for one patient: 30%*30%
        expected_true_positives.append(prop*prop)

    # actual positives is code proportion
    setp=sum(expected_true_positives)
    scp=sum(code_proportion)
    print("Expected true positives per patient", setp) # 0.653923
    print("Expected positives per patient", scp) # 3.482510
    print("Expected sensitivity:",setp/scp) # 0.187773
    # higher than actual

    # what's the expected specificity?
    negative_code_proportion=[]
    for i in range(len(code_proportion)):
        negative_code_proportion.append(1-code_proportion[i])

    expected_true_negatives=[]
    for prop in negative_code_proportion:
        # the rate of actual negative labels times the background prediction
        expected_true_negatives.append(prop*prop)
    actual_negatives=sum(negative_code_proportion)
    true_negatives=sum(expected_true_negatives)

    # what is the Gini impurity compared to the AUC?
    print("Expected true negatives per patient", true_negatives) # 186.688
    print("Expected negatives per patient", actual_negatives) # 189.517
    print("Expected specificity:",true_negatives/actual_negatives) # 0.98507

    # what is the expected binary cross entropy with random guess?
    prop=np.asarray(code_proportion)
    info=np.log(prop)

    negat_prop=1-prop
    neginfo=np.log(negat_prop)

    expected_BCE=-prop*info-negat_prop*neginfo
    # 0.0673
    print(np.mean(expected_BCE))

    print("Really?")
    print("Question, why is the ROC not 1? Toss of coin should always be 1 right?")


def ROC_1():
    """
    What is the expected ROC given background prediction?
    We have 15667 patients in training set.
    Calculate the average proportion of the code. See the chance of predicting by backround.
    This is like Gini Impurity? We will see.
    :return:
    """
    ig=InputGenJ(no_underlying=True, death_only=True)
    death_count=ig.death_code_count
    death_sort=ig.death_code_sort
    death_dict=ig.death_code_dict
    total_patients=ig.death.index.get_level_values('rep_person_id').nunique()

    code_proportion=list(death_count.values())

    for i in range(len(code_proportion)):
        code_proportion[i]/=total_patients

    expected_true_positives=[]
    for prop in code_proportion:
        # a code has 30% chance happening
        # 30% background guess
        # true positive exepctation for one patient: 30%*30%
        expected_true_positives.append(prop*prop)

    # actual positives is code proportion
    setp=sum(expected_true_positives)
    scp=sum(code_proportion)
    print("Expected true positives per patient", setp)
    print("Expected positives per patient", scp)
    print("Expected sensitivity:",setp/scp)
    # higher than actual

    # what's the expected specificity?
    negative_code_proportion=[]
    for i in range(len(code_proportion)):
        negative_code_proportion.append(1-code_proportion[i])

    expected_true_negatives=[]
    for prop in negative_code_proportion:
        # the rate of actual negative labels times the background prediction
        expected_true_negatives.append(prop*prop)
    actual_negatives=sum(negative_code_proportion)
    true_negatives=sum(expected_true_negatives)

    # what is the Gini impurity compared to the AUC?
    print("Expected true negatives per patient", true_negatives)
    print("Expected negatives per patient", actual_negatives)
    print("Expected specificity:",true_negatives/actual_negatives)

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

if __name__ == '__main__':
    calcexpcod()
