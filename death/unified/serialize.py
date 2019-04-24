# Serialize patient data for faster loading.

from death.post.inputgen_planJ import *

def serialize(path):
    print("This function will serialize your data at", path)
    print("This function will run quite a while, because it needs to collect every data point")
    cache_them(path)

if __name__ == '__main__':
    serialize()