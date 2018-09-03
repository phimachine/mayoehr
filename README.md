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

### For pandas to PyTorch dataloader: post/inputgen_planD.py

Different models will have different input/output formats. This is the input/output generation file for plan A. See post/README.md

__getitem__() is the main function for this class, which integrates with PyTorch dataloader to enable multi-processed data loading.

The InputGen has gone through several versions. Plan A is plaine old.

Plan C splits the dataset as validation and training. It allows you to use different loss functions for different samples.
For people whose mortality outcomes is known, we will use one type of loss,
for those with only the last visit known to us, we will use another type of loss.

Plan D adjusts the proportion of records with known and unknown mortality outcomes.

### DNC model: DNC/batchDNC.py

BatchDNC is capable of training multiple sequences at a time, especially
important for batch normalization on the input. BN on weights is possible,
but it's not so esay with recurrent models. Note that batch training DNC
does not require padding sequences. Concatenation of sequences is possible
with ChannelManager object, and all autograd variable storage management is
tasks is taken over by the ChannelManager too.

You can choose to use my implementation of LSTM or the stock LSTM by pytorch.
Not sure which is faster.

#### Frankenstein

This is an older model.
The name is legacy as a result of project development contingency.
All modules are in the same file. BaBi has memory overflow, but our project does not.
Loss goes down. The model is finished.

### Training: DNC/batchtrainer.py
You should run the run.py script at the root of this project, however, since that
allows Python to find its submodules.

Just do "python run.py".

### Tacotron model: taco/model.py
and taco/smalltaco.py taco/tacotrainer.py taco/smalltacotrainer.py

Tacotron is a sequence to sequence translation model that performs extremely
well. I adopted this model to translate from medical records to mortality outcomes.

There is an ablation study on the effect of the post processing unit.
Small taco does nto hvae post processing, but achieves the same performance.

### LSTM model: baseline/lstmtrainer.py

LSTM mdoel is the baseline for all deep learning models. It is the fastest

### Results for now?

The loss of all models that I have created are around 0.0053. All loss
converges extremely fast, at around 100 training samples. This shows that
our models were not able to efficiently extract information from this
EHR dataset.

I examined the output of the small taco model. The output seems to be
stuck at particular non-zero values. It shows that the model only
captures the background probabilities, but is unable to capture input-target
relationships.

## How to reuse the project

The dataset is loaded by mainly four stages: (raw,) post-processed,
pandas, flattened, PyTorch . You can build your own project upon any
of the four stages. Check reuse_examples.py for some code examples.

### I want to use the pre processed csv files by R
The preprocessed R data objects are located on infodev1 drives. The path is
/infodev1/rep/projects/jason. See /death/README.md for the file names and the
content of the files. Some post processing has been done, so I did not use the
csv files directly, and there might be additional columns computed later on.

### I want to use the post processed input/output by python
You can load the pandas tables that I have processed if you will be working in
Python. To do so, take a look at the death/post/dfmanager.py.

After the intialization, you should have dfmanager.dia to be the diagnosis
pandas dataframe, for example.


### I want to use 47774 dim input and 3620 dim output directly in Python
I flattened all the pandas dataframes to be a single vector of input and a
single vector of output for every single training point. This leads to a
47774 dim input and 3620 dim output. This method of flattening might not be
the optimal solution, might not work for every model, and certainly does not work
for every project goal. However, if you want
to start from here, you should take a look at death/post/inputgen_planD.py and
inputgen_planC.py. These two scripts will project each column of pandas
dataframes, depending on their types and factors, onto some section of the single
tensor.

### I want to run a deep learning model with pytorch
You should take a look at death/post/channelmanager.py and inputgen_planD. Both
objects are built to be/based on PyTorch Datasets and should be loaded to pytorch
with no problem.
