# Here are a few examples to reuse the code that I made.

### To see the pandas data frames
# Note that DFManager allows you to load from pickle file or raw csv files
# pickle loading is much faster. loading raw rebuilds pickle
#
from drug.post.dfmanager10 import DFManager
dfs = DFManager()
dfs.load_pickle(verbose=True)
# from here, you can see all dataframes as dfs' properties
# for example, this is demographics csv:
print(dfs.demo)

# if you want to load_raw and rebuild pickle files, run:
dfs.load_raw(save=True)
# making dictionary necessary for one-hot encodings
dfs.make_dictionary(verbose=True, save=True, skip=False)


### To see the inputs and outputs used by the deep learning model
from death.post.inputgen_planD import InputGenD, train_valid_split
ig = InputGenD(verbose=False)
# split to training set and validation set if you want
# it's fine if you don't do this step
train, valid = train_valid_split(ig)
# __getitem__() method is how you should access this dataset
print(train[123])


### Loading into PyTorch is trickier, because sequences don't have even lengths
# I have two solutions, one with ChannelManager and one with padded sequences
# you should see the script for BatchDNC and Tacotron respectively.