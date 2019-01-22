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
    dfn_coln=[("dia","dx_codes"),("dhos","dx_codes"),("ahos","dx_codes")]
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
#
# def parallel_base_case_collection():
#     def look_up(data_index):
#         # for data_index in tqdm([88]):
#         rpi = ig.rep_person_id[data_index]
#         i, t, lt = ig.get_by_id(rpi)
#         ret = look_up_any(ig, i, t, dfn_coln_words)
#         if ret:
#             return data_index
#
#
#     ig = InputGenG()
#     # rpi=ig.rep_person_id[88]
#     ds_index = []
#     dfn_coln_words = look_up_locations_drug_base_cases(ig)
#     res= Parallel(n_jobs=2)(delayed(look_up)(data_index)  for data_index in tqdm(range(20)))#len(ig))))
#     with open("/infodev1/rep/projects/jason/drug_users.txt", "w+") as f:
#         for data_index in ds_index:
#             f.write(str(data_index))
#
#     print("finished")

def look_up(data_index, ig, dfn_coln_words):
    # for data_index in tqdm([88]):
    # rpi = ig.rep_person_id[data_index]
    i,t,lt=ig[data_index]
    ret = look_up_any(ig, i, t, dfn_coln_words)
    if ret:
        return data_index

def look_up_for_imap(data_index):
    """
    This function cannot be run directly.
    dfn_coln_words and ig are assumed to be global variables that will be dealt with by the initializer of p.imap.
    :param data_index:
    :return:
    """

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

# this is the one that worked.
def multiprocessing_base_case_collection():

    # with mp.Pool(2,initializer) as p:
    #     r=list(p.imap(look_up, range(5)))

    # using 32 processers 3 it/s. One processer uses around 30 s/it
    with mp.Pool(32,initializer) as p:
        ret=list(tqdm(p.imap(look_up_for_imap, range(247428)), total=247428))

    ret=[i for i in ret if i is not None]
    with open("/infodev1/rep/projects/jason/drug_users.txt", "w+") as f:
        for data_index in ret:
            f.write(str(data_index)+"\n")
    print("Scanning finished")

def review_look_up(data_index, ig, dfn_coln_words):
    # for data_index in tqdm([88]):
    # rpi = ig.rep_person_id[data_index]
    i,t,lt=ig[data_index]
    ret = review_look_up_any(ig, i, t, dfn_coln_words)
    return ret

def review_results():
    ig=InputGenG(debug=False)
    dfn_coln_words=look_up_locations_drug_base_cases(ig)


    for di in interested:
        ret=review_look_up(di,ig,dfn_coln_words)
        rep_person_id=ig.rep_person_id[di]
        print("REP Person ID:", rep_person_id)
        print("%8s | %12s | %10s"%("dfn", "coln","code"))
        for dfn, coln, code in ret:
            print("%8s | %12s | %10s"%(dfn, coln, code))


def review_look_up_any(ig, input, target, dfn_coln_words, time=None, input_only=True):
    """
    This is a function that looks up whether a code has ever appeared in a patient history.
    Checks the whole code, not structure
    :param ig:
    :param input:
    :param target:
    :param dfn_coln_words: a list of dfn, coln and word you want to lookup. [(dfn1,coln1,word1), (dfn2,coln2,word2)...]
    :return:
    """
    def set_dict(dfn,coln):
        dic=ig.get_dict(dfn,coln)
        setattr(review_look_up_any,dfn + "_" + coln + "_dict",dict((v,k) for k, v in dic.items()))

    if time is not None:
        raise NotImplementedError()

    input_idx_list = []
    target_idx_list = []
    input_idx_dic={}
    target_idx_dic={}

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
            input_idx_dic[idx]=(dfn,coln,word)
        else:
            target_idx_list.append(idx)
            target_idx_dic[idx]=(dfn,coln,word)

    ii=np.asarray(input_idx_list)
    ti=np.asarray(target_idx_list)

    dfn_coln_codes_input = []
    dfn_coln_codes_death = []

    for array,indices,dfn_coln_codes,is_death in [(input, ii,dfn_coln_codes_input,False), (target,ti,dfn_coln_codes_death,True)]:
        if input_only and is_death:
            pass
        else:
            if indices.size!=0:
                if len(array.shape) == 1:
                    time_slice = array[indices]
                else:
                    time_slice = array[:, indices]
                # the goal is to decipher where the flag was raised.

                flag_indices=time_slice.nonzero()[1]
                # starting from here I modify the function to get information about the look_up.
                if is_death:
                    try:
                        dic = getattr(review_look_up_any, dfn + "_" + coln + "_dict")
                    except AttributeError:
                        set_dict(dfn, coln)
                        dic = getattr(review_look_up_any, dfn + "_" + coln + "_dict")
                    dfn_coln_codes.append((dfn, coln, dic[idx]))
                else:
                    for idx in flag_indices:
                        ig_loc=indices[idx]
                        for dfn, coln in ig.all_dfn_coln:
                            if dfn!="death":
                                st,end=ig.get_column_index_range(dfn,coln)
                                if st<=ig_loc and end>=ig_loc:
                                    # then this is the dfn and coln
                                    try:
                                        dic=getattr(review_look_up_any,dfn + "_" + coln + "_dict")
                                    except AttributeError:
                                        set_dict(dfn,coln)
                                        dic=getattr(review_look_up_any,dfn + "_" + coln + "_dict")

                                    no_code=True
                                    code=dic[ig_loc-st]
                                    for a,b,c in dfn_coln_words:
                                        if c==code:
                                            no_code=False
                                            break
                                    if no_code:
                                        raise ValueError("This code is not found in dfn_coln_words")
                                    dfn_coln_codes.append((dfn,coln,code))


    if input_only:
        return dfn_coln_codes_input
    else:
        return dfn_coln_codes_input,dfn_coln_codes_death

def pandas_lookup(index, ig, drug_list):
    # because the numpy lookup has significant performance issue, I wnat to try another route.
    # Maybe faster maybe slower, let's see.

    # in ig, line by line examination of relevant datasets and columns.
    # deal with bar separations
    # if true, record.

    # [("dia","dx_codes"),("dhos","dx_codes"),("ahos","dx_codes")]
    # all of them are bar separated

    # ig and drug_related_i9 should be allocated as global env var in multiprocessing.

    dia=ig.dia
    dhos=ig.dhos
    ahos=ig.ahos
    dfs=[dia,dhos,ahos]

    for df in dfs:
        try:
            rows=df.loc[[index]]
            dx_codes=rows["dx_codes"]
            for dx_code in dx_codes:
                splitted=dx_code.split("|")
                for code in splitted:
                    if code in drug_list:
                      return index
        except KeyError:
            pass

def imap_pandas_lookup(index):
    # because the numpy lookup has significant performance issue, I wnat to try another route.
    # Maybe faster maybe slower, let's see.

    # in ig, line by line examination of relevant datasets and columns.
    # deal with bar separations
    # if true, record.

    # [("dia","dx_codes"),("dhos","dx_codes"),("ahos","dx_codes")]
    # all of them are bar separated

    # ig and drug_related_i9 should be allocated as global env var in multiprocessing.

    dia=ig.dia
    dhos=ig.dhos
    ahos=ig.ahos
    dfs=[dia,dhos,ahos]

    for df in dfs:
        try:
            rows=df.loc[[index]]
            dx_codes=rows["dx_codes"]
            for dx_code in dx_codes:
                splitted=dx_code.split("|")
                for code in splitted:
                    if code in drug_list:
                      return index
        except KeyError:
            pass

def pandas_initializer():
    global ig
    global drug_list
    ig=InputGenG()
    drug_list=[]
    with open("/infodev1/rep/projects/jason/drugsi9.csv",'r') as code_file:
        for line in code_file:
            drug_list.append(line.strip())

# this is the one that worked.
def multiprocessing_pandas_base_case_collection():

    # with mp.Pool(2,initializer) as p:
    #     r=list(p.imap(look_up, range(5)))

    # using 32 processers 3 it/s. One processer uses around 30 s/it
    with mp.Pool(32,pandas_initializer) as p:
        ret=list(tqdm(p.imap(imap_pandas_lookup, range(247428)), total=247428))

    ret=[i for i in ret if i is not None]
    with open("/infodev1/rep/projects/jason/drug_users.txt", "w+") as f:
        for data_index in ret:
            f.write(str(data_index)+"\n")
    print("Scanning finished")


def some_tests_pandas():
    ig=InputGenG()
    drug_list=[]
    with open("/infodev1/rep/projects/jason/drugsi9.csv",'r') as code_file:
        for line in code_file:
            drug_list.append(line.strip())
    cnt=0
    for i in range(593,594):
        cnt+=1
        ret=pandas_lookup(i,ig,drug_list)
        if cnt==20:
            cnt=0
            print("another 20")
        if ret is not None:
            print(ret)
            print("we done")

def main():
    multiprocessing_pandas_base_case_collection()

if __name__ == '__main__':
    multiprocessing_pandas_base_case_collection()
