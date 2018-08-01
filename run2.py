# from death.DNC.trainerD2 import *
from death.DNC.notmysamtrainer import *
import os
from shutil import copy
import traceback
import datetime

if __name__ == "__main__":

    with torch.cuda.device(0):
        try:
            main()
        except:
            traceback.print_exc()