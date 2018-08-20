# from death.DNC.trainerD2 import *
from death.DNC.trashcan.notmysamtrainer import *
import traceback

if __name__ == "__main__":

    with torch.cuda.device(0):
        try:
            main()
        except:
            traceback.print_exc()