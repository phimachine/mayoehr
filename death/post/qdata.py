# This is a script unique to our project
# Much of the training/test data will need to be genereatd at run time, and this file takes care of the querying
# of all data files and return an input/label that can be fed into our model.
# Our end goal is to wrap our data object into a Torch Dataloader()

# numpy genfromtxt is extremely slow
import numpy as np
import pandas as pd
import pdb

def load_all(verbose=True):
    v=verbose

    # DEATHTARGETS
    dtypes= {'rep_person_id': "int",
             "death_date": "str",
             "underlying": "bool",
             "code": "str"}
    parse_dates= ["death_date"]
    death=pd.read_csv('/infodev1/rep/projects/jason/deathtargets.csv',dtype=dtypes)

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
    dyptes={'rep_person_id':"int",
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
            "lab_loinc_code":"str",
            "lab_abn_flag":"category",
            "smaller":"float",
            "bigger":"float"
            }
    pres=pd.read_csv("/infodev1/rep/projects/jason/mypres.csv",dtype=dtypes)
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
        print("total memory usage: "+total_mem.item()+" gb")
        print("script finished")

