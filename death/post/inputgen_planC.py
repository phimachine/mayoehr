from death.post.dfmanager import *
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import copy
import time
import torch
import collections




# we can only assume that all deaths are recorded

# torch integration reference: https://github.com/utkuozbulak/pytorch-custom-dataset-examples

# This is the plan C.

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
        # TODO we need to exploit the structured codes and augment inputs
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
        self.check_nan()
        self.debug=debug
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
        if word!=word or word=="":
            print("a code is empty or NAN")
        else:
            # start idx plus dic index
            codelist=indices[1]
            for idx in range(len(codelist)):
                codelist[idx]+=dic[word]
            np.add.at(array,indices,1)

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
                insidx = dic[val] + startidx
                np.add.at(input, [tss, insidx], 1)
        if (input != input).any():
            raise ValueError("NA FOUND")

        coln = "male"
        insidx, endidx = self.get_column_index_range(dfn, coln)
        val = row[coln].iloc[0]
        if val == val:
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

        target = np.zeros((time_length, self.output_dim))
        if id in df.index:

            # death time to event
            allrows = self.death.loc[[id]]
            death_date = allrows["death_date"].iloc[0]
            earliest_distance = (death_date.to_datetime64() - earliest.to_datetime64()).astype("timedelta64[M]").astype(
                "int")
            countdown_val = np.arange(earliest_distance, earliest_distance - time_length, -1)
            np.add.at(target, [tss, 0], countdown_val)

            # cause of death
            cods = allrows["code"]
            unds = allrows["underlying"]
            insidx = []

            for code, underlying in zip(cods, unds):
                # no na testing, I tested it in R
                # if cod==cod and und==und:
                dic = self.__getattribute__("death_code_dict")
                # try:
                #     idx = dic[code]
                # except KeyError:
                #     print("Death code does not exist")
                #     idx = dic["0"]
                self.code_into_array_structurally(target, [tss, [1]], code, dic, debug)
                if underlying:
                    self.code_into_array_structurally(target, [tss, [self.underlying_code_location]], code, dic, debug)
                # insidx += [1 + idx]
                # if underlying:
                #     insidx += [self.underlying_code_location + idx]
            # does not accumulate!
            # target[:, insidx] = 1
            loss_type = np.zeros(1,dtype=np.long)
        else:
            loss_type = np.ones(1,dtype=np.long)
            countdown_val = np.arange(time_length - 1, -1, -1)
            np.add.at(target, [tss, 0], countdown_val)
            # TODO do not pass gradient to code if loss_type is 1

        # TODO use bottom layer bias to offset missing data.

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
                barsep = []
                nobarsep = []
                for coln in datacolns:
                    if (dfn, coln) in self.no_bar:
                        nobarsep.append(coln)
                    elif (dfn, coln) in self.bar_separated:
                        barsep.append(coln)
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
                for coln in nobarsep:
                    startidx, endidx = self.get_column_index_range(dfn, coln)
                    dic = self.__getattribute__(dfn + "_" + coln + "_dict")
                    insidx = []
                    nantsloc = []

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

                for coln in barsep:
                    startidx, endidx = self.get_column_index_range(dfn, coln)
                    dic = self.__getattribute__(dfn + "_" + coln + "_dict")
                    tss = []
                    insidx = []

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

        if (input != input).any():
            raise ValueError("NA FOUND")
        # high frequency visitors have been handled smoothly, by aggregating
        if debug:
            print("get item finished")
        if (input != input).any():
            raise ValueError("NA FOUND")
        return input.astype("float32"), target.astype("float32"), loss_type

    def __len__(self):
        '''
        Length of the demographics dataset
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

class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping=mapping
        self.length=length
        self.mother=mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(ds, split_fold=10, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    This is just a factory method, nothing special. Can't believe no one ever did this for PyTorch.

    :return:
    '''
    if random_seed!=None:
        np.random.seed(random_seed)

    dslen=len(ds)
    indices= list(range(dslen))
    valid_size=dslen//split_fold
    np.random.shuffle(indices)
    train_mapping=indices[valid_size:]
    valid_mapping=indices[:valid_size]
    train=GenHelper(ds, dslen - valid_size, train_mapping)
    valid=GenHelper(ds, valid_size, valid_mapping)

    return train, valid

def oldmain():
    ig = InputGen(load_pickle=True, verbose=False)
    ig.performance_probe()

    # go get one of the values and see if you can trace it all the way back to raw data
    # this is a MUST DO TODO
    print('parallelize')
    dl = DataLoader(dataset=ig, batch_size=1, shuffle=False, num_workers=16)

    start=time.time()
    n=0
    for i,t,l, in dl:
        print(i,t,l)
        # if you take a look at the shape you will know that dl expanded a batch dim
        n+=1
        if n==100:
            break
        if (i!=i).any() or (t!=t).any():
            raise ValueError("NA found")
    end=time.time()
    print("100 rounds, including initiation:", end-start)
    # batch data loading seems to be a problem since patients have different lenghts of data.
    # it's advisable to load one at a time.
    # we need to think about how to make batch processing possible.
    # or maybe not, if the input dimension is so high.
    # well, if we don't have batch, then we don't have batch normalization.
    print("script finished")


if __name__ == "__main__":
    ig = InputGen()
    ig.__getitem__(20730)
