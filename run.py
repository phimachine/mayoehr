from death.baseline.lstmtrainer import *
import os
from shutil import copy
import traceback
import datetime

if __name__ == "__main__":

    with torch.cuda.device(1):
        main()