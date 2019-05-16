import pandas as pd
from death.unified.uni import *

if __name__ == '__main__':
    with torch.cuda.device(1):
        main4()