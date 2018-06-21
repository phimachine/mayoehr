This is a code repository for all-mortality prediction.
This is a prototying project aimed to help me explore the dataset, processing and modelling process, to gauge the challenges ahead of me.

This is a private repository. If you encounter this repository anywhere, please contact me through email: hu.yaojie@mayo.edu


## Data pre processing with R
### Dataset selection
First thing is to identify what columns I will need. Not all are relevant. It's good to select only those that are relevant, so we can reduce model complexity and chance of overfitting, and for efficient computing.

* Throw away diagnosis that are not in I9.
* Be careful with the dirty data. fread() took in those that do not have matching columns. Some years are 0202, some dxcode is alphabetical text.

1. select demographics file
2. join dataset with corresponding columns
3. remove redundancies

### Code processing and unification
Some columns have different code types. HIC/I9
I want to use I10, and I want to limit the length of codes to one decimal place.

### Death targets
Death targets need to be manually produced.

### Dataset file merging and splitting
The idea is to merge the dataset to be a huge chunk, with all relevant columns. We will not need onee-hot encoding in the dataset, to reduce harddrive I/O. In my experience, hard drive I/O takes 20% of total time. One-hot encoding will be computed at runtime.

A complete patient history is a unit of data. Sort by time within a patient. Shuffled among patients.

* We should explore the possibility of variable lenghts. Split the dataset to spanning 10 years, 5 years or 1 year. Know that pytorch supports dynamic graph, and the only real reason we have maximum story length is because we want to run a parallel of 64. I would not rule out the possibility that I can modify maximum story length for each batch to feed. This reduces the end padding.

Merged dataset needs to be split by 10G each for faster disk I/O. The whole dataset would fit in our crazy memory, but I should not do that. This means each dataset# needs to be pre-shuffled, with each patient as a unit.

## Data runtime processing with Python

Note that much of these are model-independent. I must keep these codes separately from the model.

### Loading
When we load a dataset#, we have a bunch of patients with different health records. The main difference is sparsity and length. Some patients visit often. Some patients stayed here for a short period.
This is very much what our original dataset looks like.
The data should be a table, index by patient number and datetime.

### Longitudinal conversion
As a simple solution, I will convert our longitudinal data to time series. This might not be better than using longitudinal data right away. We will experiement.
A span of a month would be reasonable. This choice is very arbitrary and not optimal. At the moment I am just looking for the easiest solution possible. I hope the loss converges.

### Disease codes' structural exploitation
I9 or ICD10 codes are structured reference numbers. E.g. 192.e. These numbers need to be processed to reflect the strcutures. We can do this by splitting it by decimal places, or we can look up in the database and find the categories for one-hot encoding.

### Load by batch.
All signals spanned from 1995 to 2016. This means if we take the whole sequence in, then we have a fixed sequence length of 20x12=240, and some signals have longer lengths.
* Why does story length even matter? Can't I have a variable length that gets fed in and evaluated normally? Interval matters much more.

### Encoding
There must be other encodings that I need to convert at run time.


## Model
### Evaluation metric
There are at least two ways to evaluate the model's performance.
1. We can evaluate at each input timestamp all future timestamps equally. Evaluating distant future data points do not introduce bias, but the variance is significant and might hinder the convergence of the model.
2. We can evaluate at the next timestamp only, if the patient comes at this point, given all past visits. This requires the model to predict where data does not exist, much like our bAbI story problem.

A critical question to consider is whether the metric would bias towards patient who visit the hospital more often. If the model biasedly thinks that patients in this dataset comes to hospital very often, then for patients who come only once, the model will predict higher illness rate than normal.

If we use method 2, then we are saying that the model needs to predict the next encounter. What if the there are two illnesses alternating? The model will never be able to predict correctly. Both methods have problems.

It seems that method 2 has less convergence, but method 1 predict more targets, which naturally speeds up convergence too.

### Evaluation metric
I decided to combine the two metrics together. The network has to produce two goals at the same time.
First, will he die eventually? Binary label that pertains to a sequence.
Second, will he die right now? Binary label that pertains to a timestamp (next timestamp). Last timestamp assumes that he is not dead.
This is reasonable for our project. Other goals will not use the same, but will be pretty similar

### debugging
Somehow the model's loss does not converge in the end. It's possible that somewhere there is a bug in the architecture that needs to be addressed. I should look into that.
