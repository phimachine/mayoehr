import pandas as pd
from death.unified.uni import *

if __name__ == '__main__':
    # test_all_for_AUROC_1(epoch=40, test_stat_dir="40epoch")
    # test_all_for_AUROC_1(epoch=20, test_stat_dir="20epoch")
    test_all_for_AUROC_1(epoch=4, test_stat_dir="4epoch")

    # transformer201()