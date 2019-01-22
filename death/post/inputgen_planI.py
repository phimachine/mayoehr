# Our goal is to learn mortality outcomes. Our training data must be previously known mortality outcomes, therefore,
# Learning from patients whose mortality outcomes are unknown and coerce the goal to be zero vector is unreasonable.
# This is unreasonable for training, testing and validation.
# This is unreasonable for sequence based prediction only. Ideally, we want a timestep based prediction, and if that's
# the case, patients who are alive at some timestep does yield a goal probability that's zero for all causes of death.
# But we are not going into that plan anymore.
# This only applies in our motality studies. For drug studies, this does not apply.

from death.post.dfmanager import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from numpy.random import permutation
import time
import torch
from multiprocessing.pool import ThreadPool as Pool
import os
import traceback


def get_timestep_location(earliest, dates):
    '''
    Uses numpy instead of panda.
    dates must be month based, otherwise errors will arise.

    :param earliest: pandas.Timestamp
    :param dates: ndarray of datetime64
    :return: cc: int numpy array, the index location of the corresponding records
    '''
    dates=dates.apply(lambda x: x.replace(day=1))
    earliest = earliest.to_datetime64()
    # if not isinstance(time,pd.Timestamp):
    # # if it's a series as it should be
    #     time=time.values
    # else:
    #     time=time.to_datetime64()
    dates = dates.values
    cc = (dates - earliest).astype('timedelta64[M]')
    return cc.astype("int")


# multiple inheritance
class InputGen(Dataset, DFManager):
    '''
    This is the second object in the python data generation pipeline
    Take a data frame manager object and produce inputs wrapped in torch objects through Dataset interface
    See get_by_id()
    '''

    def __init__(self, verbose=False, debug=False):
        super(InputGen, self).__init__()
        self.load_pickle(verbose=verbose)
        self.rep_person_id = self.demo.index.values
        self.verbose = verbose
        # 47774
        self.input_dim = None
        # manual format: (dfname,colname,starting_index)
        self.input_dim_manual = None
        self.output_dim = None
        self.get_input_dim()
        self.get_output_dim()
        # this df has no na
        self.earla = pd.read_csv("/infodev1/rep/projects/jason/earla.csv", parse_dates=["earliest", "latest"])
        self.earla.set_index("rep_person_id", inplace=True)
        self.len = len(self.rep_person_id)
        # self.check_nan()
        self.debug=debug
        if self.verbose:
            print("Input Gen initiated")

    def get_output_dim(self):
        '''
        The output is death_date and cause of death, respectively [:,0] and [:,-0]
        :return:
        '''
        # dimsize (death_date,cause)
        dimsize = 1 + 1
        dic = self.__getattribute__("death_code_dict")
        dimsize += 2 * len(dic)
        self.output_dim = dimsize
        self.underlying_code_location = 1 + len(dic)

        return dimsize

    def get_input_dim(self):
        # pre allocate a whole vector of input
        input_dim_manual = []
        dimsize = 0
        for dfn in self.dfn:
            if dfn!="death":
                df = getattr(self, dfn)
                # get all columns and column dtypes, allocate depending on the dtypes
                for colname, dtype in zip(df.dtypes.index, df.dtypes):
                    if colname == "rep_person_id" or self.is_date_column(colname):
                        # no memory needed for these values.
                        # either index that is ignored, or contained in the time series.
                        if dfn == "demo" and self.is_date_column(colname):
                            # then we are dealing with birth date
                            input_dim_manual.append((dfn, colname, dimsize))
                            # how many dimensions do you want for birth date?
                            # My plan is to simply throw the age in as a float.
                            dimsize += 1
                    else:
                        dtn = dtype.name
                        input_dim_manual.append((dfn, colname, dimsize))
                        if self.verbose:
                            print("accounting for", dfn, colname)
                        if dtn == 'bool':
                            dimsize += 1
                        elif dtn == "category":
                            dimsize += len(self.get_dict(dfn, colname))
                        elif dtn == "object":
                            dimsize += len(self.get_dict(dfn, colname))
                        elif dtn == "float64":
                            dimsize += 1
                        elif dtn == "int64":
                            dimsize += 1
                        elif dtn == "datetime64[ns]":
                            raise ValueError("No, I should not see this")
                        else:
                            raise ValueError("Unaccounted for")

        self.input_dim = dimsize
        self.input_dim_manual = input_dim_manual

    def get_column_index_range(self, dfn, coln):
        '''
        standard notation [start,end)
        modifies self.input_dim_manual and self.input_base_size
        :param dfn:
        :param coln:
        :return: start, end: integer, memory index
        '''
        idx = 0
        start = None
        end = None
        while (idx < len(self.input_dim_manual)):
            if self.input_dim_manual[idx][0] == dfn and self.input_dim_manual[idx][1] == coln:
                start = self.input_dim_manual[idx][2]
                break
            idx += 1
        idx += 1
        if idx < len(self.input_dim_manual):
            end = self.input_dim_manual[idx][2]
        else:
            end = self.input_dim

        return start, end

    def __getitem__(self, index):
        '''

        :param index:
        :return:
        '''
        id = self.rep_person_id[index]
        return self.get_by_id(id,self.debug)

    def code_into_array_structurally(self, array, indices, word, dic, debug):
        """
        Lookup the dictionary and insert the code by it structure into the array.

        :param array:
        :param indices: [timesteps, startindices], both of them must be lists.
        :param values:
        :return:
        """
        # type checking. np.add.at behaves differently for different types
        if debug:
            if not isinstance(indices[0],collections.Iterable) or not isinstance(indices[1], collections.Iterable):
                raise TypeError("code_into_array_structurally() type error, indices[n] must be iterable")
        assert (len(indices[1])==1)
        if word not in ("", "empty", "None", "none"):
            # start idx plus dic index
            codelist=indices[1]
            struct_code_list=[]
            while word !="":
                struct_code_list.append(dic[word]+codelist[0])
                word=word[:-1]
            indices[1]=struct_code_list
            np.add.at(array, indices, 1)

    def get_by_id(self,id,debug=False):
        time_length = self.earla.loc[id]["int"] + 1
        earliest = self.earla.loc[id]["earliest"]
        latest = self.earla.loc[id]["latest"]
        input = np.zeros((time_length, self.input_dim), dtype=float)

        ######
        # We start compiling input and target.
        # demo

        row = self.demo.loc[[id]]
        tss = np.arange(time_length)
        dfn = "demo"
        for coln in ("race", "educ_level"):
            startidx, endidx = self.get_column_index_range(dfn, coln)
            dic = self.__getattribute__(dfn + "_" + coln + "_dict")

            # I know that only one row is possible
            val = row[coln].iloc[0]
            if val == val:
                if val not in ("", "empty", "None", "none"):
                    insidx = dic[val] + startidx
                    np.add.at(input, [tss, insidx], 1)
        if (input != input).any():
            raise ValueError("NA FOUND")

        coln = "male"
        insidx, endidx = self.get_column_index_range(dfn, coln)
        val = row[coln].iloc[0]
        if val == val:
            if val not in ("", "empty", "None", "none"):
                if val:
                    np.add.at(input, [tss, insidx], 1)
        # this might have problem, we should use two dimensions for bool. But for now, let's not go back to prep.
        if (input != input).any():
            raise ValueError("NA FOUND")
        coln = "birth_date"
        insidx, _ = self.get_column_index_range(dfn, coln)
        bd = row[coln].iloc[0]
        if bd == bd:
            # convert age
            earliest_month_age = (earliest.to_datetime64() - bd.to_datetime64()).astype("timedelta64[M]").astype("int")
            age_val = np.arange(earliest_month_age, earliest_month_age + time_length)
            np.add.at(input, [tss, insidx], age_val)
        if (input != input).any():
            raise ValueError("NA FOUND")
        #####
        # death
        # we need time_to_event, cause of death, and loss type
        df = self.death

        # now no longer changes according to time
        target = np.zeros((1, self.output_dim))
        if id in df.index:

            # death time to event
            allrows = self.death.loc[[id]]
            death_date = allrows["death_date"].iloc[0]
            # now change to latest distance
            latest_distance = (death_date.to_datetime64() - latest.to_datetime64()).astype("timedelta64[M]").astype(
                "int")
            # countdown_val = np.arange(earliest_distance, earliest_distance - time_length, -1)
            np.add.at(target, (0, 0), latest_distance)

            # cause of death
            cods = allrows["code"]
            unds = allrows["underlying"]

            for code, underlying in zip(cods, unds):
                # no na testing, I tested it in R
                # if cod==cod and und==und:
                dic = self.__getattribute__("death_code_dict")
                # try:
                #     idx = dic[code]
                # except KeyError:
                #     print("Death code does not exist")
                #     idx = dic["0"]
                self.code_into_array_structurally(target, [[0], [1]], code, dic, debug)
                if underlying:
                    self.code_into_array_structurally(target, [[0], [self.underlying_code_location]], code, dic, debug)
                # insidx += [1 + idx]
                # if underlying:
                #     insidx += [self.underlying_code_location + idx]
            # does not accumulate!
            # target[:, insidx] = 1
            loss_type = np.zeros(1,dtype=np.long)
        else:
            loss_type = np.ones(1,dtype=np.long)
            # countdown_val now is zero, so no need to add
            # countdown_val = np.arange(time_length - 1, -1, -1)
            # np.add.at(target, [tss, 0], countdown_val)
        target=target.squeeze(0)

        #####
        # all other dataframes, will insert at specific timestamps
        others = [dfn for dfn in self.dfn if dfn not in ("death", "demo")]
        for dfn in others:
            # any df is processed here
            df = self.__getattribute__(dfn)
            if id in df.index:
                allrows = df.loc[[id]]

                # get the index for all dates first
                date_coln = [coln for coln in df if self.is_date_column(coln)]

                if debug:
                    assert len(date_coln) == 1
                datacolns = [coln for coln in df if
                             not self.is_date_column(coln) and coln not in ("rep_person_id", "id")]
                date_coln = date_coln[0]

                # I hate that the return value of this line is inconsistent
                # If single value it's timestamp, if multiple it's np time list
                all_dates = allrows[date_coln]
                tsloc = get_timestep_location(earliest, all_dates)

                # we bucket the columns so we know how to process them.
                direct_insert = []
                barsepi9 = []
                barsepcate = []
                nobari9 = []
                cate=[]
                for coln in datacolns:
                    if (dfn, coln) in self.no_bar_i9:
                        nobari9.append(coln)
                    elif (dfn, coln) in self.bar_separated_i9:
                        barsepi9.append(coln)
                    elif (dfn, coln) in self.categories:
                        cate.append(coln)
                    elif (dfn, coln) in self.bar_separated_categories:
                        barsepcate.append(coln)
                    else:
                        direct_insert.append(coln)
                        if debug:
                            try:
                                assert (self.dtypes[dfn][coln] in ("int", "bool", "float"))
                            except (KeyError, AssertionError):
                                raise

                # we need two things: index and values
                for coln in direct_insert:
                    startidx, endidx = self.get_column_index_range(dfn, coln)
                    if debug:
                        try:
                            assert (endidx - startidx == 1)
                        except AssertionError:
                            raise
                    # this line will increment only 1:
                    # input[tsloc,startidx]+=allrows[coln]
                    try:
                        np.add.at(input, (tsloc, startidx), np.nan_to_num(allrows[coln].values))
                    except IndexError:
                        print("we found it")
                if (input != input).any():
                    raise ValueError("NA FOUND")

                # codes
                for coln in cate:
                    startidx, _ = self.get_column_index_range(dfn, coln)
                    dic = self.__getattribute__(dfn + "_" + coln + "_dict")
                    for ts, val in zip(tsloc, allrows[coln]):
                        if val not in ("", "empty", "None", "none"):
                            codeidx=dic[val]
                            np.add.at(input,[[ts],[codeidx+startidx]], 1)

                for coln in nobari9:
                    startidx, endidx = self.get_column_index_range(dfn, coln)
                    dic = self.__getattribute__(dfn + "_" + coln + "_dict")

                    for ts, val in zip(tsloc, allrows[coln]):
                        # if not nan
                        self.code_into_array_structurally(input,[[ts],[startidx]],val,dic, debug)
                    #     if val == val and val != "":
                    #         insidx += [dic[val] + startidx]
                    #         nantsloc += [ts]
                    # np.add.at(input, [nantsloc, insidx], 1)
                    # again, accumulate count if multiple occurrence
                if (input != input).any():
                    raise ValueError("NA FOUND")

                for coln in barsepi9:
                    startidx, endidx = self.get_column_index_range(dfn, coln)
                    dic = self.__getattribute__(dfn + "_" + coln + "_dict")

                    for ts, multival in zip(tsloc, allrows[coln]):
                        if multival == multival:
                            vals = multival.split("|")
                            vals = list(filter(lambda a: a != "", vals))
                            vals = list(filter(lambda a: a != "empty", vals))

                            for val in vals:
                                self.code_into_array_structurally(input,[[ts],[startidx]],val,dic, debug)
                            #
                            # tss += [ts] * len(vals)
                            # insidx += [dic[val] + startidx for val in vals if val == val]
                    # try:
                    #     np.add.at(input, [tss, insidx], 1)
                    # except IndexError:
                    #     raise IndexError

                for coln in barsepcate:
                    startidx, endidx = self.get_column_index_range(dfn, coln)
                    dic = self.__getattribute__(dfn + "_" + coln + "_dict")

                    for ts, multival in zip(tsloc, allrows[coln]):
                        if multival == multival:
                            vals = multival.split("|")
                            vals = list(filter(lambda a: a != "", vals))
                            vals = list(filter(lambda a: a != "empty", vals))

                            for val in vals:
                                if val not in ("", "empty", "None", "none"):
                                    np.add.at(input,[[ts],[startidx+dic[val]]],1)


        if (input != input).any():
            raise ValueError("NA FOUND")
        # high frequency visitors have been handled smoothly, by aggregating
        if debug:
            print("get item finished")
        if (input != input).any():
            raise ValueError("NA FOUND")
        return input.astype("float32"), target.astype("float32"), loss_type.astype("float32")

    def __len__(self):
        '''
        Length of the demographics dataset
        247428
        :return:
        '''
        return self.len

    def performance_probe(self):
        # all dfn have unique double index. Performance is not yet known.
        # I hope they hash hierarchically.

        # for dfn in self.dfn:
        #     if dfn!="demo":
        #         df=self.__getattribute__(dfn)
        #         print(dfn, "has unique index?", df.index.is_unique)
        #         print(dfn, "is lex_sorted?", df.index.is_lexsorted())
        #         print("....")

        start = time.time()
        for i in range(4):
            input, target, loss_type = ig.__getitem__(i, debug=True)
            print("working on ", i)
        end = time.time()
        print(end - start)
        print("performance probe finished")
        print("speed is now 3x faster")

    def check_nan(self):
        # this function checks all data frames and ensure that there is no nan anywhere.
        print("checking if there is nan in any of the dataframes")
        for dfn in self.dfn:
            df=self.__getattribute__(dfn)
            if df.isnull().values.any():
                print("NA found in dataframe", dfn)
                print(df.isna().any())

        return

# class GenHelper(Dataset):
#     def __init__(self, mother, length, mapping):
#         # here is a mapping from this index to the mother ds index
#         self.mapping=mapping
#         self.length=length
#         self.mother=mother
#
#     def __getitem__(self, index):
#         return self.mother[self.mapping[index]]
#
#     def __len__(self):
#         return self.length


class GenHelper(Dataset):
    def __init__(self, ids, ig):
        """

        :param ids: needs to be combined and randomized first
        :param ig: inputgen
        """

        self.ids=ids
        self.ig=ig
        self.small_target=ig.small_target

    def __getitem__(self, index):

        ret_val = self.ig.get_by_id(self.ids[index])
        if self.small_target:
            st=ret_val[1][:2976]
            return ret_val[0], st, ret_val[2]
        else:
            return ret_val

    def __len__(self):
        return len(self.ids)


class InputGenI(InputGen):
    def __init__(self, curriculum=False, validation_test_proportion=0.1, random_seed=54321,small_target=False,
                 debug=False,verbose=False):
        verbose = verbose
        debug = debug
        super(InputGenI, self).__init__(verbose=verbose, debug=debug)

        self.validation_test_proportion=validation_test_proportion
        self.random_seed=random_seed
        np.random.seed(random_seed)
        # for curriculum learning, every time the training set is requested, the death proportion will be adjusted
        self.curriculum=curriculum

        self.valid=None
        self.test=None
        self.train_valid_test_split()

        # the proportion of death records the last training set has
        self.proportion=None
        if verbose:
            print("Using InputGenI")
            if self.curriculum:
                print("Using curriculum learning")
            else:
                print("Not using curriculum learning")

        self.small_target=small_target

    def train_valid_test_split(self):
        # splits the whole set by id
        death_rep_person_id = self.death.index.get_level_values(0).unique().values
        death_rep_person_id=permutation(death_rep_person_id)
        valid_or_test_death_len=int(len(death_rep_person_id)*self.validation_test_proportion)
        self.valid_death_id=death_rep_person_id[:valid_or_test_death_len]
        self.test_death_id=death_rep_person_id[valid_or_test_death_len:valid_or_test_death_len*2]
        self.train_death_id=death_rep_person_id[valid_or_test_death_len*2:]

        # no_death_rep_person_id = self.rep_person_id[np.invert(np.in1d(self.rep_person_id, death_rep_person_id))]
        # no_death_rep_person_id=permutation(no_death_rep_person_id)
        # valid_or_test_no_death_len=int(len(no_death_rep_person_id)*self.validation_test_proportion)
        # self.valid_no_death_id=no_death_rep_person_id[:valid_or_test_no_death_len]
        # self.test_no_death_id=no_death_rep_person_id[valid_or_test_no_death_len:valid_or_test_no_death_len*2]
        # self.train_no_death_id=no_death_rep_person_id[valid_or_test_no_death_len*2:]

    def get_valid(self):
        """
        should be run only once in its lifetime
        :return:
        """
        if self.valid is None:
            ids=self.valid_death_id
            ids=permutation(ids)
            self.valid=GenHelper(ids, self)
        return self.valid

    def get_test(self):
        """
        should be run only once in its lifetime
        :return:
        """
        if self.test is None:
            ids=self.test_death_id
            ids=permutation(ids)
            self.test=GenHelper(ids, self)
        return self.test

    def get_train(self):
        """
        modifies the deathfold everytime it is called
        :return:
        """
        # resample_rate=2**self.death_fold
        # new_no_death_length=len(self.train_no_death_id)//resample_rate
        # new_no_death_id=permutation(self.train_no_death_id)
        # new_no_death_id=new_no_death_id[:new_no_death_length]
        # self.proportion=len(self.train_death_id)/(len(self.train_death_id)+len(new_no_death_id))
        # print("Death proportion", self.proportion, ", death fold",  self.death_fold)
        # ids=np.concatenate((self.train_death_id,new_no_death_id))
        ids=permutation(self.train_death_id)
        train=GenHelper(ids,self)
        if self.curriculum:
            self.change_fold()
        return train

    def change_fold(self):
        if self.death_fold==0:
            print("Death fold is zero, death fold not changed")
        else:
            self.death_fold-=1

    def pickle_death_code_count(self):
        # this function calculates the death structure codes's frequency
        # for a code, number_of_negatives/number_of_positives will weigh the positive binary cross entropy
        # this allows me to sample more points on the ROC

        dic=self.death_code_dict
        weights=np.zeros((len(dic)))
        for row in self.death["code"]:
            code=row
            while code!="":
                weights[dic[code]]+=1
                code=code[:-1]

        with open("/infodev1/rep/projects/jason/pickle/dcc.pkl", "wb+") as f:
            pickle.dump(weights,f)




def pad_sequence(sequences, batch_first=False, padding_value=0):
    r"""Pad a list of variable length Tensors with zero

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *`` where `T` is the
            length of the longest sequence.
        Function assumes trailing dimensions and type of all the Tensors
            in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if batch_first is False
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


def pad_collate(args):
    val=list(zip(*args))
    tensors=[[torch.from_numpy(arr) for arr in vv] for vv in val]
    padded=[pad_sequence(ten) for ten in tensors]
    padded[0]=padded[0].permute(1,0,2)
    padded[1]=padded[1].permute(1,0)
    return padded

# def selective_cache_train_valid():
#     ig=InputGenG(death_fold=0)
#     ig.train_valid_test_split()
#     valid=ig.get_valid()
#     valid_cacher=DatasetCacher("zerofold/valid",valid)
#     valid_cacher.cache_some(16,500)
#
#
#     print("Done")

# def igtest():
#     ig=InputGenI(death_fold=0)
#     ig.train_valid_test_split()
#     valid=ig.get_valid()
#     test=ig.get_test()
#     train=ig.get_train()
#     print(valid[100])
#     print(train[192])
#     cacher=DatasetCacher("test/npz",valid)
#     for i in range(10):
#         cacher.cache_one(i)
#     print(cacher[55])
#     print("Done")
#     print(cacher[2])
#
#     for i in range(10):
#         input, target, l = cacher[i]
#         print(target[0])

# def main():
#     selective_cache()

if __name__=="__main__":
    ig=InputGenI()
    for idx in range(10):
        i,t,l=ig[idx]
        print(i,t,l)
    print("DONe")