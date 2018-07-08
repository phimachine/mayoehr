# This is a script unique to our project
# Much of the training/test data will need to be genereatd at run time, and this file takes care of the querying
# of all data files and return an input/label that can be fed into our model.
# Our end goal is to wrap our data object into a Torch Dataloader()

# numpy genfromtxt is extremely slow
import numpy as np
import pandas as pd

dtypes={"rep_person_id":'int',
"SRV_DATE":'str',
'srv_px_count':'int',
'srv_px_code':'str',
'SRV_LOCATION':'str',
'srv_admit_type':'str',
'srv_admit_src':'str',
'srv_disch_stat':'str'}
parse_dates=["SRV_DATE"]
pd.read_csv('/infodev1/rep/projects/jason/myserv.csv',dtype=dtypes,parse_dates=parse_dates)
