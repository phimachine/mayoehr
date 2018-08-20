# this file reads all post processed R .csv files and put them in pandas
# you need to run the main function every time you change any .csv file to update the pickled pandas
# dataframes.
# usually you do not need to modify anything in this script and pickling will be automatic
# it will take a few minutes.

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import collections

pickle_path = "/infodev1/rep/projects/jason/pickle/"


class DFManager(object):
    '''
    Dataframe manager
    It is the first object in the data generation pipelines in Python.
    It takes csv files and make pandas dataframes, then pickle them.
    '''

    def __init__(self):
        '''
        Dataframes, along with some useful properties
        '''
        self.loaded = False

        # not in order
        # these information will be used for automation purposes across the whole project
        # they contain possible human errors, and must be kept prestine

        # do not contain all columns of the datasets
        self.categories = [("demo", "educ_level"), ("dhos", "hosp_adm_source"), ("dhos", "hosp_disch_disp"),
                           ("ahos", "hosp_adm_source"), ("ahos", "hosp_disch_disp"),
                           ("lab", "lab_abn_flag"), ("demo", "race"), ("serv", "SRV_LOCATION"),
                           ("serv", "srv_admit_type"),
                           ("serv", "srv_admit_src"), ("serv", "srv_disch_stat")]
        self.bar_separated = [("dia", "dx_codes"), ("dhos", "dx_codes"),("ahos", "dx_codes"),
                              ("pres", "med_ingr_rxnorm_code")]
        self.no_bar = self.categories + [("death", "code"), ("lab", "lab_loinc_code"), ("serv", "srv_px_code"),
                                         ("surg", "collapsed_px_code")]


        self.dtypes = collections.OrderedDict()
        self.dtypes["death"] = collections.OrderedDict([('rep_person_id', "int"),
                                                        ("death_date", "str"),
                                                        ("underlying", "bool"),
                                                        ("code", "str")])
        self.dtypes["demo"] = collections.OrderedDict([('rep_person_id', "int"),
                                                       ('race', "category"),
                                                       ("educ_level", "category"),
                                                       ("birth_date", "str"),
                                                       ("male", "bool")])
        self.dtypes["dia"] = collections.OrderedDict([('rep_person_id', "int"),
                                                      ('dx_date', "str"),
                                                      ("dx_codes", "str")])
        self.dtypes["ahos"] = collections.OrderedDict([("rep_person_id", "int"),
                                                      ("hosp_admit_dt", "str"),
                                                      ("hosp_adm_source", "category"),
                                                      ("hosp_disch_disp", "category"),
                                                      ("dx_codes", "str"),
                                                      ("is_in_patient", "bool")])
        self.dtypes["dhos"] = collections.OrderedDict([("rep_person_id", "int"),
                                                      ("hosp_disch_dt", "str"),
                                                      ("hosp_adm_source", "category"),
                                                      ("hosp_disch_disp", "category"),
                                                      ("dx_codes", "str"),
                                                      ("is_in_patient", "bool")])
        self.dtypes["lab"] = collections.OrderedDict([("rep_person_id", "int"),
                                                      ("lab_date", "str"),
                                                      ("lab_loinc_code", "str"),
                                                      ("lab_abn_flag", "category"),
                                                      ("smaller", "float"),
                                                      ("bigger", "float")])
        self.dtypes["pres"] = collections.OrderedDict([("rep_person_id", "int"),
                                                       ("MED_DATE", "str"),
                                                       ("med_ingr_rxnorm_code", "str")])
        self.dtypes["serv"] = collections.OrderedDict([("rep_person_id", 'int'),
                                                       ("SRV_DATE", 'str'),
                                                       ('srv_px_count', 'int'),
                                                       ('srv_px_code', 'str'),
                                                       ('SRV_LOCATION', 'category'),
                                                       ('srv_admit_type', 'category'),
                                                       ('srv_admit_src', 'category'),
                                                       ('srv_disch_stat', 'category')])
        self.dtypes["surg"] = collections.OrderedDict([("rep_person_id", "int"),
                                                       ("px_date", "str"),
                                                       ("collapsed_px_code", "str")])
        self.dtypes["vital"] = collections.OrderedDict([("rep_person_id", "int"),
                                                         ("VITAL_DATE", "str"),
                                                         ("BMI", "float"),
                                                         ("BP DIASTOLIC", "float"),
                                                         ("BP SYSTOLIC", 'float'),
                                                         ("HEIGHT", "float"),
                                                         ("WEIGHT", "float")])

        self.fpaths={
            "death":"/infodev1/rep/projects/jason/newdeath.csv",
            "demo":"/infodev1/rep/projects/jason/demo.csv",
            "dia":"/infodev1/rep/projects/jason/multidia.csv",
            "ahos":"/infodev1/rep/projects/jason/multiahos.csv",
            "dhos": "/infodev1/rep/projects/jason/multidhos.csv",
            "lab" :"/infodev1/rep/projects/jason/multilab.csv",
            "pres" :"/infodev1/rep/projects/jason/multipres.csv",
            "serv" : '/infodev1/rep/projects/jason/multiserv.csv',
            'surg' : "/infodev1/rep/projects/jason/multisurg.csv",
            "vital": "/infodev1/rep/projects/jason/multivital.csv"
        }

        self.dfn=tuple(self.dtypes.keys())
        for dfn in self.dfn:
            setattr(self,dfn,None)
        for df, col in self.bar_separated:
            setattr(self, df + "_" + col + "_dict", None)
        for df, col in self.no_bar:
            setattr(self, df + "_" + col + "_dict", None)

    def get_dict(self, df, col):
        return getattr(self, df + "_" + col + "_dict")

    def fill_na(self, dfn):
        df=self.__getattribute__(dfn)
        filldict={}
        for key,value in self.dtypes[dfn].items():
            if value == "int":
                filldict[key]=0
            elif value == "float":
                filldict[key]=0
            elif value == "bool":
                filldict[key]=False
            elif value == "str":
                filldict[key]=""
            elif value == "category":
                df[key].cat.add_categories(["None"], inplace=True)
                filldict[key]="None"
            else:
                raise ValueError("A fill na value that is not managed")
        df.fillna(filldict,inplace=True)

    def load_raw(self, verbose=True, save=True):

        '''
        load all preprocessed datasets, return in the order of death,demo,dia,hos,lab,pres,serv,surg,vital

        :param verbose: verbose output
        :param save:
        :return: death,demo,dia,hos,lab,pres,serv,surg,vital: panda dataframe objects
        '''
        if not verbose:
            print("loading from raw, this process might take 5 minutes")

        if not self.loaded:
            v = verbose

            for dfn in self.dfn:
                dtypes=self.dtypes[dfn]
                parse_dates=[coln for coln in dtypes.keys() if self.is_date_column(coln)]
                df=pd.read_csv(self.fpaths[dfn], dtype=dtypes, parse_dates=parse_dates)
                self.__setattr__(dfn,df)
                if dfn!="demo":
                    df.set_index(["rep_person_id","id"],inplace=True)
                else:
                    df.set_index(["rep_person_id"],inplace=True)
                if v:
                    print(dfn+":")
                    print(self.__getattribute__(dfn))
                df.sort_index()
                self.fill_na(dfn)

            self.loaded = True

        if save:
            mypath = Path("/infodev1/rep/projects/jason/pickle/pddfs.pkl")

            pickle.dump(tuple(self.__getattribute__(dfn) for dfn in self.dfn),mypath.open('wb'))

            print("pickled all dfs at", mypath)

        return tuple(self.__getattribute__(dfn) for dfn in self.dfn)

    def is_date_column(self,colname):
        if "dt" in colname.lower() or "date" in colname.lower():
            return True
        else:
            return False

    def load_pickle(self, verbose=False):
        try:
            # load df
            print("Loading dataframes from pickle file")
            with open("/infodev1/rep/projects/jason/pickle/pddfs.pkl", 'rb') as f:
                self.death, self.demo, self.dia, self.ahos, self.dhos, self.lab, self.pres, \
                self.serv, self.surg, self.vital = pickle.load(f)

                self.loaded = True

            for df, col in self.bar_separated + self.no_bar:
                if verbose:
                    print("loading dictionary on bar separated " + df + " " + col)
                savepath = Path(pickle_path) / "dicts" / (df + "_" + col + ".pkl")
                with savepath.open('rb') as f:
                    dic = pickle.load(f)
                    self.__setattr__(df + "_" + col + "_dict", dic)

        except (OSError, IOError) as e:
            raise FileNotFoundError("pickle pddfs not found")

    def make_dictionary(self, save=True, verbose=False, skip=True):
        '''
        Collects all codes in all data sets, and produces dictionary for one-hot
        encoding purposes, word to index.

        This is so that when a rep_person_id is pulled, a row from panda dfs can be turned into
        numpy arrays.

        This is done exhaustively through every row of every set. Dictionaries are pickled and
        saved in Dfs.

        :param write:
        :return:
        '''

        if self.loaded == False:
            try:
                self.load_pickle()
                if verbose:
                    print('pickle file not found, loading from raw')
            except FileNotFoundError:
                self.load_raw(verbose=verbose)

        print("file loaded, making dictionaries")

        for df, col in self.bar_separated:
            if verbose:
                print("generating dictionary on bar separated " + df + " " + col)
            self.bar_separated_dictionary(df, col, save=save, skip=skip)

        for df, col in self.no_bar:
            if verbose:
                print("generating dictionary on no bar " + df + " " + col)
            self.no_bar_dictionary(df, col, save=save, skip=skip)

    def bar_separated_dictionary(self, df_name, col_name, save=True, skip=True):
        '''

        :param df_name:
        :param col_name:
        :param save:
        :return:
        '''

        savepath = Path(pickle_path) / "dicts" / (df_name + "_" + col_name + ".pkl")
        print(savepath)

        if skip:
            try:
                with savepath.open('rb') as f:
                    dic = pickle.load(f)
                    setattr(self, df_name + "_" + col_name + "_dict", dic)
                    return dic
            except FileNotFoundError:
                pass

        dic = {}
        n = 0

        series = self.get_series(df_name, col_name)

        for row in series:
            try:
                if not pd.isna(row):
                    splitted = row.split("|")
                    for word in splitted:
                        # Is there empty? in case I forgot in R
                        if not pd.isna(word):
                            if word not in dic and word != "":
                                if word == "NA":
                                    print("unprocessed NA FOUND")
                                dic[word] = n
                                n += 1

            except AttributeError:
                print("woah woah woah what did you do")
                raise

        if save == True:
            with savepath.open("wb") as f:
                pickle.dump(dic, f, protocol=pickle.HIGHEST_PROTOCOL)
                print('saved')
        getattr(self, df_name + "_" + col_name + "_dict", dic)

        return dic

    def no_bar_dictionary(self, df_name, col_name, save=True, skip=True):

        savepath = Path(pickle_path) / "dicts" / (df_name + "_" + col_name + ".pkl")
        print("save to path:", savepath)

        if skip:
            try:
                with savepath.open('rb') as f:
                    dic = pickle.load(f)
                    setattr(self, df_name + "_" + col_name + "_dict", dic)
                    return dic
            except FileNotFoundError:
                pass
        dic = {}
        n = 0

        series = self.get_series(df_name, col_name)

        for row in series:
            if not pd.isna(row):
                if row not in dic and row != "":
                    if row == "NA":
                        print("unprocessed NA FOUND")
                    dic[row] = n
                    n += 1

        if save == True:
            with savepath.open('wb') as f:
                pickle.dump(dic, f, protocol=pickle.HIGHEST_PROTOCOL)
                print('saved')
        getattr(self, df_name + "_" + col_name + "_dict", dic)

        return dic

    def get_series(self, df_name, col_name):
        '''
        Helps with batch processing with list of string df names and col names
        :param df_name: 
        :param col_name: 
        :return: 
        '''''
        return self.__getattribute__(df_name)[col_name]

def repickle():
    '''
    Run this function if you need to remake the pickled panda dataframes
    This function will take 30 minutes to 1 hour to run, mainly disk I/O.
    '''

    dfs = DFManager()
    dfs.load_raw(save=True)
    # dfs.load_pickle()
    dfs.make_dictionary(verbose=True,save=True,skip=False)
    print("end script")



if __name__ == "__main__":
    repickle()