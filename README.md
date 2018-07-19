# mayoehr

## Where are the main files?
### For preprocessing: prep/prep.R
Preprocessing takes the REP data that I was given, cleans, imputes, mutates for machine learning purposes.

I have not run the whole script myself, but the it should take around 5 hours. There are likely lines I did not record in the file, so you should do it block by block to be safe.

The whole script requires up to 32+ cores and at least 200Gb RAM at peak by my estimate.

### For postprocessing: post/post.R
Takes preprocessed csv files and transform them to meet python-specific needs. For example, one date per row, or multiindex for pandas performance.

The whole script will take around 10 minutes, <10 cores, and ~20Gb RAM.

### For R to Python pandas: post/qdata.py

Load the postprocessed csv files and turn them into pandas object. The __main__ method will pickle all the python objects and create dictionaries.

The whole script should take around 5 minutes. You shoud run it everytime you modify the csv files: edit the data file description in the DFManager object and the script should take care of the rest.

### For pandas to PyTorch dataloader: post/inputgen_planA.py

Different models will have different input/output formats. This is the input/output generation file for plan A. See post/README.md

__getitem__() is the main function for this class, which will integrate seamlessly (hopefully) with PyTorch dataloader to enable multi-processed data loading.

### DNC model: DNC/frankenstein.py
The name is legacy as a result of project development contingency.
All modules are in the same file. BaBi has memory overflow, but our project does not.
Loss goes down. The model is finished.