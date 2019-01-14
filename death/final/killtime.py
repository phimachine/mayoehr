# this script runs a function and kills it when time conditions are met
#
from death.final.killtime import *
import multiprocessing
import time
import torch.multiprocessing as mp

import datetime
class NotRightNow(Exception):
    pass


def out_of_time():
    now=datetime.datetime.now()
    if  now.hour>=8 and now.hour<=17 and not (now.weekday() in (5,6)) :
        print(now.hour, now.minute)
        print("Time satisfied")
        raise NotRightNow

def time_dec(func):
    def func_that_cares_about_time(*args, **kwargs):
        print("Will terminate the process given time")
        # multiprocessing.set_start_method("spawn")
        p=multiprocessing.Process(target=func,name="main",args=args, kwargs=kwargs)
        p.start()
        try:
            while True:
                out_of_time()
                # every 10 minutes, check the time.
                time.sleep(10)
        except NotRightNow:
            if p.is_alive():
                print("Terminating process")
                p.terminate()
                p.join(10)
    return func_that_cares_about_time