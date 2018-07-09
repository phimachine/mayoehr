# This is a script unique to our project
# Much of the training/test data will need to be genereatd at run time, and this file takes care of the querying
# of all data files and return an input/label that can be fed into our model.
# Our end goal is to wrap our data object into a Torch Dataloader()

# numpy genfromtxt is extremely slow
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

class Dfs():

    def __init__(self):
        '''
        Dataframes, along with some useful properties
        '''
        self.loaded=False

        self.death=None
        self.demo=None
        self.dia=None
        self.hos=None
        self.lab=None
        self.pres=None
        self.serv=None
        self.surg=None
        self.vital=None

        self.bar_separated=[("dia","dx_codes"),("hos","dx_codes"),
                       ("pres","med_ingr_rxnorm_code"), ("serv","srv_px_code")]
        self.no_bar=[("death","code"),("labs","lab_loinc_code"), ("serv","srv_px_code"),
                ("surg","collapsed_px_code")]

        for df, col in self.bar_separated:
            setattr(self,df+"_"+col+"_dict",None)
        for df, col in self.no_bar:
            setattr(self, df + "_" + col + "_dict", None)

    def load_raw(self, coerce=False, verbose=True, write_pickle=True):

        '''
        load all preprocessed datasets, return in the order of death,demo,dia,hos,lab,pres,serv,surg,vital

        :param coerce: coerce load and resave pickle files
        :param write_pickle:
        :param verbose: verbose output
        :return: death,demo,dia,hos,lab,pres,serv,surg,vital: panda dataframe objects
        '''
        if not verbose:
            print("loading from raw, this process might take 5 minutes")

        if not self.loaded or (self.loaded and coerce):
            v=verbose

            # DEATHTARGETS
            dtypes= {'rep_person_id': "int",
                     "death_date": "str",
                     "underlying": "bool",
                     "code": "str"}
            parse_dates= ["death_date"]
            death=pd.read_csv('/infodev1/rep/projects/jason/deathtargets.csv',dtype=dtypes,parse_dates=parse_dates)

            if v:
                print (death)
            # DEMOGRAPHICS
            dtypes={'rep_person_id': "int",
                    'race':"category",
                    "educ_level":"category",
                    "male":"bool"}
            demo=pd.read_csv("/infodev1/rep/projects/jason/demo.csv",dtype=dtypes)
            if v:
                print(demo)

            # DIAGNOSIS
            dtypes={'rep_person_id':"int",
                    'dx_date':"str",
                    "dx_codes":"str"}
            parse_dates=["dx_date"]
            dia=pd.read_csv("/infodev1/rep/projects/jason/mydia.csv",dtype=dtypes,parse_dates=parse_dates)
            if v:
                print(dia)

            # Hospitalization
            dtypes={"rep_person_id": "int",
                    "hosp_admit_dt":"str",
                    "hosp_disch_dt":"str",
                    "hosp_adm_source":"category",
                    "hosp_disch_disp":"category",
                    "dx_codes":"str",
                    "is_in_patient":"bool"}
            parse_dates=["hosp_admit_dt","hosp_disch_dt"]
            hos=pd.read_csv("/infodev1/rep/projects/jason/myhosp.csv",dtype=dtypes,parse_dates=parse_dates)
            if v:
                print(hos)

            # Labs
            dtypes={"rep_person_id": "int",
                    "lab_loinc_code":"str",
                    "lab_abn_flag":"category",
                    "smaller":"float",
                    "bigger":"float"
                    }
            lab=pd.read_csv("/infodev1/rep/projects/jason/mylabs.csv",dtype=dtypes)
            if v:
                print(lab)

            # Prescription
            dtypes={"rep_person_id": "int",
                    "MED_DATE":"str",
                    "med_ingr_rxnorm_code":"str",
                    }
            parse_dates=["MED_DATE"]
            pres=pd.read_csv("/infodev1/rep/projects/jason/mypres.csv",dtype=dtypes,parse_dates=parse_dates)
            if v:
                print(pres)

            # Services
            # 1% of memory, one of the bigger files. This is looking good.
            dtypes = {"rep_person_id": 'int',
                      "SRV_DATE": 'str',
                      'srv_px_count': 'int',
                      'srv_px_code': 'str',
                      'SRV_LOCATION': 'str',
                      'srv_admit_type': 'str',
                      'srv_admit_src': 'str',
                      'srv_disch_stat': 'str'}
            parse_dates = ["SRV_DATE"]
            serv=pd.read_csv('/infodev1/rep/projects/jason/myserv.csv', dtype=dtypes, parse_dates=parse_dates)
            if v:
                print(serv)

            # surgeries
            dtypes={"rep_person_id": "int",
                    "px_date":"str",
                    "px_code":"int",
                    "collapsed_px_code":"int"}
            parse_dates=["px_date"]
            surg=pd.read_csv("/infodev1/rep/projects/jason/mysurg.csv",dtype=dtypes,parse_dates=parse_dates)
            if v:
                print(surg)

            # vitals
            dtypes={"rep_person_id": "int",
                    "VITAL_DATE":"str",
                    "BMI":"float",
                    "BP DIASTOLIC":"float",
                    "BP SYSTOLIC":'float',
                    "HEIGHT":"float",
                    "WEIGHT":"float"}
            parse_dates=["VITAL_DATE"]
            vital=pd.read_csv("/infodev1/rep/projects/jason/myvitals.csv",dtype=dtypes,parse_dates=parse_dates)
            if v:
                print(vital)

            dfs=[death,demo,dia,hos,lab,pres,serv,surg,vital]
            total_mem=0
            for df in dfs:
                total_mem+=df.memory_usage().sum()
            total_mem=total_mem/1024/1024/1024
            if verbose:
                # ~ 10.76 gb
                print("total memory usage: ",total_mem.item()," gb")

            self.death=death
            self.demo=demo
            self.dia=dia
            self.hos=hos
            self.lab=lab
            self.pres=pres
            self.serv=serv
            self.surg=surg
            self.vital=vital

            self.loaded=True

        if write_pickle:
            mypath=Path("/infodev1/rep/projects/jason/pickle/pddfs.pkl")

            pickle.dump((self.death, self.demo, self.dia, self.hos, self.lab, self.pres, \
                               self.serv, self.surg, self.vital), mypath.open("wb"))

        return self.death, self.demo, self.dia, self.hos, self.lab, self.pres, \
               self.serv, self.surg, self.vital

    def load_pickle(self):
        try:
            with open("/infodev1/rep/projects/jason/pickle/pddfs.pkl",'rb') as f:
                self.death, self.demo, self.dia, self.hos, self.lab, self.pres, \
                self.serv, self.surg, self.vital= pickle.load(f)
                self.loaded=True
                print("Loaded from pickle file")
        except (OSError, IOError) as e:
            raise FileNotFoundError("pickle pddfs not found")

    def make_dictionary(self, write=False,verbose=False):
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

        if self.loaded==False:
            try:
                self.load_pickle()
                if verbose:
                    print('pickle file not found, loading from raw')
            except FileNotFoundError:
                self.load_raw(verbose=verbose)

        if verbose:
            print("file loaded, making dictionaries")


        for df,col in self.bar_separated:
            if verbose:
                print("generating dictionary on "+df+" "+col)
            self.bar_separated_dictionary(df,col)

        for df,col in self.no_bar:
            if verbose:
                print("generating dictionary on " + df + " " + col)
            self.no_bar_dictionary(df,col)


    def bar_separated_dictionary(self,df_name,col_name,save=False):
        '''

        :param df_name:
        :param col_name:
        :param save:
        :return:
        '''

        dic={}
        n=0

        series=self.get_series(df_name,col_name)

        for row in series:
            splitted=row.split("|")
            for word in splitted:
                # Is there empty? in case I forgot in R
                if word not in dic and word!="":
                    if word=="NA":
                        print("NA FOUND")
                    dic[word]=n
                    n+=1

        if save==True:
            savepath=Path().absolute()/"dicts"/df_name+"_"+col_name+".pkl"
            with savepath.open as f:
                pickle.dump(dic,f,protocol=pickle.HIGHEST_PROTOCOL)
        getattr(self,df_name+"_"+col_name+"_dict",dic)

        return dic

    def no_bar_dictionary(self,df_name,col_name,save=False):

        dic={}
        n=0

        series=self.get_series(df_name,col_name)

        for row in series:
            if row not in dic and row!="":
                if row=="NA":
                    print("NA FOUND")
                dic[row]=n
                n+=1

        if save==True:
            savepath=Path().absolute()/"dicts"/df_name+"_"+col_name+".pkl"
            with savepath.open as f:
                pickle.dump(dic,f,protocol=pickle.HIGHEST_PROTOCOL)
        getattr(self,df_name+"_"+col_name+"_dict",dic)

        return dic

    def get_series(self,df_name,col_name):
        '''
        Helps with batch processing with list of string df names and col names
        :param df_name: 
        :param col_name: 
        :return: 
        '''''
        return self.__getattribute__(df_name)[col_name]

if __name__=="__main__":
    dfs=Dfs()
    # dfs.load_raw()
    dfs.make_dictionary(verbose=False)