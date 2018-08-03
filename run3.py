from death.DNC.trainerD2 import *
# from death.DNC.notmysamtrainer import *
import traceback
import datetime

if __name__ == "__main__":
    with torch.cuda.device(1):
        print("Using the second CUDA device")
        print("Salvage mode, will attempt to save the most recent weights you have")
        try:
            forevermain(True, 1e-3, savestr="pal",palette=True)
        except:
            traceback.print_exc()
            with open("error.log", 'a') as f:
                f.write(str(datetime.datetime.now().time()))
                traceback.print_exc(file=f)
            salvage("pal")
