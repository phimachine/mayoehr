from death.post.inputgen_planJ import InputGenJ
import pandas as pd
from collections import Counter
import tqdm
import numpy as np

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

if __name__ == '__main__':
    ROC_2()