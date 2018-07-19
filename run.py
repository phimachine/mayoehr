from death.DNC.trainerCF import *
import os
from shutil import copy


if __name__ == "__main__":
    print("Salvage mode")
    try:
        main()
    except:
        salvage()
