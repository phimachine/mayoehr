from death.DNC.trainerD import *
import os
from shutil import copy
import traceback
import datetime

if __name__ == "__main__":
    print("Salvage mode, will attempt to save the most recent weights you have")
    try:
        main()
    except:
        traceback.print_exc()
        with open("error.log", 'a') as f:
            f.write(str(datetime.datetime.now().time()))
            traceback.print_exc(file=f)
        salvage()
