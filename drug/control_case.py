from death.post.inputgen_planG import InputGenG
import numpy as np
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
import multiprocessing as mp
import pathos
import pathos.pools as pp

def what_df_contains_drug_code():
    ig=InputGenG()
    for df, coln in ig.bar_separated_i9 + ig.no_bar:
        dict=ig.get_dict(df,coln)
        sample_codes=["9610","9095","V5889","9090","E959","E8502","E9397"]
        cnt=0
        for code in sample_codes:
            if code in dict:
                cnt+=1
        print("Dictionary",df, coln,"has",cnt,"codes")

    """
    Dictionary dia dx_codes has 7 codes
    Dictionary dhos dx_codes has 7 codes
    Dictionary ahos dx_codes has 7 codes
    Dictionary pres med_ingr_rxnorm_code has 0 codes
    Dictionary demo educ_level has 0 codes
    Dictionary dhos hosp_adm_source has 0 codes
    Dictionary dhos hosp_disch_disp has 0 codes
    Dictionary ahos hosp_adm_source has 0 codes
    Dictionary ahos hosp_disch_disp has 0 codes
    Dictionary lab lab_abn_flag has 0 codes
    Dictionary demo race has 0 codes
    Dictionary serv SRV_LOCATION has 0 codes
    Dictionary serv srv_admit_type has 0 codes
    Dictionary serv srv_admit_src has 0 codes
    Dictionary serv srv_disch_stat has 0 codes
    Dictionary death code has 0 codes
    Dictionary lab lab_loinc_code has 0 codes
    Dictionary serv srv_px_code has 2 codes
    Dictionary surg collapsed_px_code has 0 codes
    """



def look_up_any(ig, input, target, dfn_coln_words, time=None):
    """
    This is a function that looks up whether a code has ever appeared in a patient history.
    Checks the whole code, not structure
    :param ig:
    :param input:
    :param target:
    :param dfn_coln_words: a list of dfn, coln and word you want to lookup. [(dfn1,coln1,word1), (dfn2,coln2,word2)...]
    :return:
    """
    if time is not None:
        raise NotImplementedError()

    input_idx_list = []
    target_idx_list = []

    for dfn, coln, word in dfn_coln_words:
        if dfn != "death":
            # then array is input
            startidx, endidx = ig.get_column_index_range(dfn, coln)

            # if word not in ("", "empty", "None", "none"):
            #     dic = ig.get_dict(dfn, coln)
            #     wordidx = dic[word]
            #     idx = wordidx + startidx
            #     input_idx_list.append(idx)
        else:
            # then array is target
            assert (coln == "code")
            startidx = 1

        dic = ig.get_dict(dfn, coln)
        if word not in ("", "empty", "None", "none"):
            try:
                wordidx = dic[word]
            except KeyError:
                print("The word not consulted is", word, "in", dfn, coln)
        idx = wordidx + startidx

        if dfn != "death":
            input_idx_list.append(idx)
        else:
            target_idx_list.append(idx)

    ii=np.asarray(input_idx_list)
    ti=np.asarray(target_idx_list)

    flag=False
    for array,indices in [(input, ii), (target,ti)]:
        if indices.size!=0:
            if len(array.shape) == 1:
                time_slice = array[indices]
            else:
                time_slice = array[:, indices]
            if (time_slice == 1).any():
                flag=True

    return flag

def look_up_locations_drug_base_cases(ig):
    drug_list=[]
    with open("/infodev1/rep/projects/jason/drugsi9.csv",'r') as code_file:
        for line in code_file:
            drug_list.append(line.strip())
    dfn_coln=[("dia","dx_codes"),("dhos","dx_codes"),("ahos","dx_codes"),("serv","srv_px_code")]
    dfn_coln_words=[]
    for dfn, coln in dfn_coln:
        for drug in drug_list:
            dic=ig.get_dict(dfn,coln)
            try:
                wordidx=dic[drug]
                dfn_coln_words.append((dfn,coln,drug))
            except KeyError:
                pass

    return dfn_coln_words

def base_case_collection():
    ig=InputGenG()
    # rpi=ig.rep_person_id[88]
    ds_index=[]
    dfn_coln_words=look_up_locations_drug_base_cases(ig)
    for data_index in tqdm(range(len(ig))):
    # for data_index in tqdm([88]):
        rpi=ig.rep_person_id[data_index]
        i, t, lt = ig.get_by_id(rpi)
        ret = look_up_any(ig, i, t, dfn_coln_words)
        if ret:
            ds_index.append(data_index)
    with open("/infodev1/rep/projects/jason/drug_users.txt","w+") as f:
        for data_index in ds_index:
            f.write(str(data_index))

    print("finished")

def parallel_base_case_collection():
    def look_up(data_index):
        # for data_index in tqdm([88]):
        rpi = ig.rep_person_id[data_index]
        i, t, lt = ig.get_by_id(rpi)
        ret = look_up_any(ig, i, t, dfn_coln_words)
        if ret:
            return data_index


    ig = InputGenG()
    # rpi=ig.rep_person_id[88]
    ds_index = []
    dfn_coln_words = look_up_locations_drug_base_cases(ig)
    res= Parallel(n_jobs=2)(delayed(look_up)(data_index)  for data_index in tqdm(range(20)))#len(ig))))
    with open("/infodev1/rep/projects/jason/drug_users.txt", "w+") as f:
        for data_index in ds_index:
            f.write(str(data_index))

    print("finished")


def look_up(data_index):

    # for data_index in tqdm([88]):
    # rpi = ig.rep_person_id[data_index]
    i,t,lt=ig[data_index]
    ret = look_up_any(ig, i, t, dfn_coln_words)
    if ret:
        return data_index

def initializer():
    global ig
    global dfn_coln_words
    ig=InputGenG(verbose=False)
    dfn_coln_words=look_up_locations_drug_base_cases(ig)
#
# def multiprocessing_base_case_collection():
#
#
#     ig = InputGenG()
#     # rpi=ig.rep_person_id[88]
#     ds_index = []
#     dfn_coln_words = look_up_locations_drug_base_cases(ig)
#
#     # def setup(ig,dfn_coln_words):
#     #     global inputgg
#     #     global dcw
#     #     inputgg=ig
#     #     dcw=dfn_coln_words
#
#
#
#     # with Pool(2) as p:
#     #     r=list(tqdm(p.map(partial(look_up,ig=ig,dfn_coln_words=dfn_coln_words),range(20)),total=20))
#     #
#     # res= Parallel(n_jobs=2)(delayed(look_up)(data_index)  for data_index in tqdm(range(20)))#len(ig))))
#     # with Pool(2) as p:
#     #     res = list(tqdm(p.starmap(look_up,[(di,ig,dfn_coln_words) for di in range(20)]),total=20))
#
#     # pool= pp.ProcessPool(2)
#     # for _ in tqdm(pool.imap_unordered(look_up,range(20),[ig]*20,[dfn_coln_words]*20),total=20):
#     #     pass
#     #     # r=tqdm(p.imap(look_up,range(20),[ig]*20,[dfn_coln_words]*20),total=20)
#
#     # this never runs to an end.
#     # for i, _ in enumerate(pool.imap(look_up,range(20),[ig]*20,[dfn_coln_words]*20), 1):
#     #     print('\rdone {0:%}'.format(i / 20))
#
#     with open("/infodev1/rep/projects/jason/drug_users.txt", "w+") as f:
#         for data_index in ds_index:
#             f.write(str(data_index))
#
#     print("finished")



def some_tests():
    ig=InputGenG(debug=False)
    i,t,lt=super(InputGenG,ig).__getitem__(88)
    ret=look_up_any(ig, t, "death", "code", ["41189","41189"])


    print(ret)
    print("hello")

def parallel_test():
    from math import sqrt
    result = Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in tqdm(range(100000)))

#
# from multiprocessing import Pool
# import tqdm
# import time
#
#
# def _foo(my_number):
#     square = my_number * my_number
#     time.sleep(1)
#     return square
#
#
# def mutlipro_test():
#
#     with Pool(2) as p:
#         r = list(tqdm.tqdm(p.imap(_foo, range(30)), total=30))

def multiprocessing_base_case_collection():

    # with mp.Pool(2,initializer) as p:
    #     r=list(p.imap(look_up, range(5)))

    # using 32 processers 3 it/s. One processer uses around 30 s/it
    with mp.Pool(32,initializer) as p:
        ret=list(tqdm(p.imap(look_up, range(247428)),total=247428))

    ret=[i for i in ret if i is not None]
    with open("/infodev1/rep/projects/jason/drug_users.txt", "w+") as f:
        for data_index in ret:
            f.write(str(data_index)+"\n")
    print("Scanning finished")

def main():
    multiprocessing_base_case_collection()

if __name__ == '__main__':
    main()
