# This is a script unique to our project
# Much of the training/test data will need to be genereatd at run time, and this file takes care of the querying
# of all data files and return an input/label that can be fed into our model.
# Our end goal is to wrap our data object into a Torch Dataloader()

# numpy genfromtxt is extremely slow
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import collections

pickle_path = "/infodev1/rep/projects/jason/pickle/"


class DFManager(object):
    '''
    Dataframe manager
    '''

    def __init__(self):
        '''
        Dataframes, along with some useful properties
        '''
        self.loaded = False

        self.death = None
        self.demo = None
        self.dia = None
        self.ahos = None
        self.dhos=None
        self.lab = None
        self.pres = None
        self.serv = None
        self.surg = None
        self.vital = None


        # not in order
        self.categories = [("demo", "educ_level"), ("dhos", "hosp_adm_source"), ("ahos", "hosp_disch_disp"),
                           ("dhos", "hosp_adm_source"), ("ahos", "hosp_disch_disp"),
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
                                                       ("px_code", "int"),
                                                       ("collapsed_px_code", "int")])
        self.dtypes["vital"] = collections.OrderedDict([("rep_person_id", "int"),
                                                         ("VITAL_DATE", "str"),
                                                         ("BMI", "float"),
                                                         ("BP DIASTOLIC", "float"),
                                                         ("BP SYSTOLIC", 'float'),
                                                         ("HEIGHT", "float"),
                                                         ("WEIGHT", "float")])

        self.fpaths={
            "death":"/infodev1/rep/projects/jason/deathtargets.csv",
            "demo":"/infodev1/rep/projects/jason/demo.csv",
            "dia":"/infodev1/rep/projects/jason/mydia.csv",
            "ahos":"/infodev1/rep/projects/jason/admit_hos.csv",
            "dhos": "/infodev1/rep/projects/jason/disch_hos.csv",
            "lab" :"/infodev1/rep/projects/jason/mylabs.csv",
            "pres" :"/infodev1/rep/projects/jason/mypres.csv",
            "serv" : '/infodev1/rep/projects/jason/myserv.csv',
            'surg' : "/infodev1/rep/projects/jason/mysurg.csv",
            "vital": "/infodev1/rep/projects/jason/myvitals.csv"
        }

        self.dfn=list(self.dtypes.keys())

        for df, col in self.bar_separated:
            setattr(self, df + "_" + col + "_dict", None)
        for df, col in self.no_bar:
            setattr(self, df + "_" + col + "_dict", None)

    def get_dict(self, df, col):
        return getattr(self, df + "_" + col + "_dict")

    def load_raw(self, coerce=False, verbose=True, save=True):

        '''
        load all preprocessed datasets, return in the order of death,demo,dia,hos,lab,pres,serv,surg,vital

        :param coerce: coerce load and resave pickle files
        :param verbose: verbose output
        :param save:
        :return: death,demo,dia,hos,lab,pres,serv,surg,vital: panda dataframe objects
        '''
        if not verbose:
            print("loading from raw, this process might take 5 minutes")

        if not self.loaded or (self.loaded and coerce):
            v = verbose

            for dfn in self.dfn:
                dtypes=self.dtypes[dfn]
                parse_dates=[coln for coln in dtypes.keys() if self.is_date_column(coln)]
                df=pd.read_csv(self.fpaths[dfn], dtype=dtypes, parse_dates=parse_dates)
                self.__setattr__(dfn,df)
                df.set_index("rep_person_id",inplace=True)
                if v:
                    print(self.__getattribute__(dfn))
            self.loaded = True

        if save:
            mypath = Path("/infodev1/rep/projects/jason/pickle/pddfs.pkl")

            pickle.dump(tuple(self.__getattribute__(dfn) for dfn in self.dfn),mypath.open('wb'))

            print("pickled all dfs")

        return tuple(self.__getattribute__(dfn) for dfn in self.dfn)
    #
    # def old_load_raw(self, coerce=False, verbose=True, save=True):
    #
    #     '''
    #     load all preprocessed datasets, return in the order of death,demo,dia,hos,lab,pres,serv,surg,vital
    #
    #     :param coerce: coerce load and resave pickle files
    #     :param verbose: verbose output
    #     :param save:
    #     :return: death,demo,dia,hos,lab,pres,serv,surg,vital: panda dataframe objects
    #     '''
    #     if not verbose:
    #         print("loading from raw, this process might take 5 minutes")
    #
    #     if not self.loaded or (self.loaded and coerce):
    #         v = verbose
    #
    #         # DEATHTARGETS
    #         dtypes = {'rep_person_id': "int",
    #                   "death_date": "str",
    #                   "underlying": "bool",
    #                   "code": "str"}
    #         parse_dates = ["death_date"]
    #         death = pd.read_csv('/infodev1/rep/projects/jason/deathtargets.csv', dtype=dtypes, parse_dates=parse_dates)
    #         death.set_index(["rep_person_id"], inplace=True)
    #         if v:
    #             print(death)
    #
    #         # DEMOGRAPHICS
    #         dtypes = {'rep_person_id': "int",
    #                   'race': "category",
    #                   "educ_level": "category",
    #                   "birth_date": "str",
    #                   "male": "bool"}
    #         parse_dates = ["birth_date"]
    #         demo = pd.read_csv("/infodev1/rep/projects/jason/demo.csv", dtype=dtypes, parse_dates=parse_dates)
    #         demo.set_index(["rep_person_id"], inplace=True)
    #         if v:
    #             print(demo)
    #
    #         # DIAGNOSIS
    #         dtypes = {'rep_person_id': "int",
    #                   'dx_date': "str",
    #                   "dx_codes": "str"}
    #         parse_dates = ["dx_date"]
    #         dia = pd.read_csv("/infodev1/rep/projects/jason/mydia.csv", dtype=dtypes, parse_dates=parse_dates)
    #         dia.set_index(["rep_person_id"], inplace=True)
    #         if v:
    #             print(dia)
    #
    #         # Hospitalization
    #         dtypes = {"rep_person_id": "int",
    #                   "hosp_admit_dt": "str",
    #                   "hosp_disch_dt": "str",
    #                   "hosp_adm_source": "category",
    #                   "hosp_disch_disp": "category",
    #                   "dx_codes": "str",
    #                   "is_in_patient": "bool"}
    #         parse_dates = ["hosp_admit_dt", "hosp_disch_dt"]
    #         hos = pd.read_csv("/infodev1/rep/projects/jason/myhosp.csv", dtype=dtypes, parse_dates=parse_dates)
    #         hos.set_index("rep_person_id", inplace=True)
    #         if v:
    #             print(hos)
    #
    #         # Labs
    #         dtypes = {"rep_person_id": "int",
    #                   "lab_date": "str",
    #                   "lab_loinc_code": "str",
    #                   "lab_abn_flag": "category",
    #                   "smaller": "float",
    #                   "bigger": "float"
    #                   }
    #         parse_dates = ["lab_date"]
    #         lab = pd.read_csv("/infodev1/rep/projects/jason/mylabs.csv", dtype=dtypes, parse_dates=parse_dates)
    #         lab.set_index("rep_person_id", inplace=True)
    #         if v:
    #             print(lab)
    #
    #         # Prescription
    #         dtypes = {"rep_person_id": "int",
    #                   "MED_DATE": "str",
    #                   "med_ingr_rxnorm_code": "str"
    #                   }
    #         parse_dates = ["MED_DATE"]
    #         pres = pd.read_csv("/infodev1/rep/projects/jason/mypres.csv", dtype=dtypes, parse_dates=parse_dates)
    #         pres.set_index("rep_person_id", inplace=True)
    #         if v:
    #             print(pres)
    #
    #         # Services
    #         # 1% of memory, one of the bigger files. This is looking good.
    #         dtypes = {"rep_person_id": 'int',
    #                   "SRV_DATE": 'str',
    #                   'srv_px_count': 'int',
    #                   'srv_px_code': 'str',
    #                   'SRV_LOCATION': 'category',
    #                   'srv_admit_type': 'category',
    #                   'srv_admit_src': 'category',
    #                   'srv_disch_stat': 'category'}
    #         parse_dates = ["SRV_DATE"]
    #         serv = pd.read_csv('/infodev1/rep/projects/jason/myserv.csv', dtype=dtypes, parse_dates=parse_dates)
    #         # should be the only double index dataset
    #         serv.set_index(['rep_person_id', 'SRV_DATE'], inplace=True)
    #         if v:
    #             print(serv)
    #
    #         # surgeries
    #         dtypes = {"rep_person_id": "int",
    #                   "px_date": "str",
    #                   "px_code": "int",
    #                   "collapsed_px_code": "int"}
    #         parse_dates = ["px_date"]
    #         surg = pd.read_csv("/infodev1/rep/projects/jason/mysurg.csv", dtype=dtypes, parse_dates=parse_dates)
    #         surg.set_index("rep_person_id", inplace=True)
    #         if v:
    #             print(surg)
    #
    #         # vitals
    #         dtypes = {"rep_person_id": "int",
    #                   "VITAL_DATE": "str",
    #                   "BMI": "float",
    #                   "BP DIASTOLIC": "float",
    #                   "BP SYSTOLIC": 'float',
    #                   "HEIGHT": "float",
    #                   "WEIGHT": "float"}
    #         parse_dates = ["VITAL_DATE"]
    #         vital = pd.read_csv("/infodev1/rep/projects/jason/myvitals.csv", dtype=dtypes, parse_dates=parse_dates)
    #         vital.set_index("rep_person_id", inplace=True)
    #         if v:
    #             print(vital)
    #
    #         dfs = [death, demo, dia, hos, lab, pres, serv, surg, vital]
    #         total_mem = 0
    #         for df in dfs:
    #             total_mem += df.memory_usage().sum()
    #         total_mem = total_mem / 1024 / 1024 / 1024
    #         if verbose:
    #             # ~ 10.76 gb
    #             print("total memory usage: ", total_mem.item(), " gb")
    #
    #         self.death = death
    #         self.demo = demo
    #         self.dia = dia
    #         self.hos = hos
    #         self.lab = lab
    #         self.pres = pres
    #         self.serv = serv
    #         self.surg = surg
    #         self.vital = vital
    #
    #         self.loaded = True
    #
    #     if save:
    #         mypath = Path("/infodev1/rep/projects/jason/pickle/pddfs.pkl")
    #
    #         pickle.dump((self.death, self.demo, self.dia, self.hos, self.lab, self.pres, \
    #                      self.serv, self.surg, self.vital), mypath.open("wb"))
    #         print("pickled all dfs")
    #
    #     return self.death, self.demo, self.dia, self.hos, self.lab, self.pres, \
    #            self.serv, self.surg, self.vital

    def is_date_column(self,colname):
        if "dt" in colname.lower() or "date" in colname.lower():
            return True
        else:
            return False

    def load_pickle(self, verbose=False):
        try:
            # load df
            print("loading from pickle file")
            with open("/infodev1/rep/projects/jason/pickle/pddfs.pkl", 'rb') as f:
                self.death, self.demo, self.dia, self.hos, self.lab, self.pres, \
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


if __name__ == "__main__":
    dfs = DFManager()
    dfs.load_raw(save=True)

    # dfs.load_pickle()

    dfs.make_dictionary(verbose=True,save=True,skip=True)

    print("end script")
