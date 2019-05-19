import pandas as pd
from death.unified.uni import *

if __name__ == '__main__':
    ds = ExperimentManager(batch_size=64, num_workers=4, total_epochs=20)
    ds.lstm_with_prior(load=False)
    ds.save_str="lstm031"
    ds.run()
