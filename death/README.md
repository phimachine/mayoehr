This is a code repository for all-mortality prediction.
This is a prototying project aimed to help me explore the dataset, processing and modelling process, to gauge the challenges ahead of me.

This is a private repository. If you encounter this repository anywhere, please contact me through email: hu.yaojie@mayo.edu


## Data pre processing with R

Note: all codes have been converted to ICD9 standard, mainly in order to reduce the dimensions.
We have two conversion files, one for diagnosis and one for procedures.
Diagnosis: 2018_I10gem.txt, from https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-CM-and-GEMs.html
Procedures: gem_pcsi9.txt, from https://www.cms.gov/Medicare/Coding/ICD10/Downloads/ProcedureGEMs-2014.zip

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

Pandas does not allow NA for int columns, but to avoid that, we can use categories, and deal with the inconsistencies with Numpy arrays later.

## Data file description
All data files are stored on /infodev1/rep/projects/jason/, the "data" folder in this repository stores public data.
The idea is to pull an id from demographics file and query all files for a complete patient record at run time.
The data is preprocessed specifically for deep learning architectures.

The post processing file might give you more details on the data file types. Just run the postproc, pause in the end and examine the pandas data frames.

9 files in total.

### Death targets
deathtargets.csv

59,394 rows 

#### Columns:
* rep_person_id: 18231 cases. sorted on as default for this project.
* death_date: date, specified to days. sorted secondarily on as default for this project. 1981-07-26 to 2017-09-30
* underlying: binary, is underlying cause of death
* code: ICD9 code for diseases at death, without dot, converted from 2018 release of I10 gem, with fuzzy match

#### note: 
* Does not contain patients who are alive.
* NA coding is NA, default for this project unless specified

### Demographics
demo.csv

250,056 rows

#### Columns:
* rep_person_id: 250056 cases, includes all rep_person_id in other files.
* male: binary label for sex. B is considered F for model simplicity.
* race: {1,2,3,4,5,6,98,99}, coding scheme unknown.
* educ_level: {0,1,2,3,4,5}, coding scheme unknown.

#### Note:
Age has been discarded in this dataset. This reduces our prediction power, but forces the model to discover from medical indicators.

### Diagnosis
mydia.csv

Files with bar separated value fields should obey this condition: every row has a unique rep_person_id, dx_date tuple.

22,886,619 rows

#### Columns:
* rep_person_id: 247246.
* dx_date: 1995-01-01 to 2016-12-31
* dx_codes: bar separated i9 code without dot, converted from 2018 release of I10 gem, with fuzzy match. those rows with empty dx_codes are filled with "empty". 12537 unique values including "empty". 2% empty before bar append

#### Notes:
* Empty is basically a flag for existing diagnosis. I trust that the model will be able to deal with it correctly. Most of the time, it seems from the original dataset, that this is just a reundant row that follows a diagnosis, but this is not always the case.

### Hospitalization
myhosp.csv

1,219,229 rows

#### Columns:
* rep_person_id: 192,049 unique
* hosp_admit_dt: admission date. 1997-01-01 to 2016-12-31
* hosp_disch_dt: discharge date. 1997-01-02 to 2017-05-06
* hosp_adm_source: 21 factors. Levels:  1 2 5 7 8 9 A B D E EL ER F G H K M N UR XXX. Coding scheme unknown. Value with less than 1000 count is combined to XXX (other).
* dx_codes: i9 code without dot, converted from 2018 release of I10 gem, with fuzzy match.
* is_inpatient: binary flag, is inpatient, otherwise outpatient.

### Labs
mylabs.csv

This is a dirty file.

77,866,916 rows

#### Columns:
* rep_person_id: 209,310 unique
* lab_date: 1995-01-01, 2016-12-31
* lab_loinc_code: the only file with this encoding, 3594 uniques
* lab_abn_flag: some sort of flag. 19 Levels:  * a AB ABN C CH CL CRIT h H HI l L LOW N P Unknown
* smaller: negative values that indicate the deviation of measurement against the prescribed range, in the unit of standard deviations, assuming that the measurement obey normal distribution.
* bigger: positive values. ditto.

#### Notes: 
* missing values of loinc code has been imputed if the lab_src_code has appeared in our database

### Prescription
new_min_mypres.csv/mypres.csv

11,203,383 rows

This file is the only dirty file in our dataset, and I have cleaned it aggressively with insufficient conditions. Given that this dataset has ~12 million rows, it's impossible to examine each clearly.
Now when I think about it, testing existence of bars might be a btter condition, since most of the errors are file I/O errors

#### Columns:
* rep_person_id: 125,969 unique
* MED_DATE: the prescription date, i.e. the starting date. 2004-01-01 to 2016-12-31
* med_ingr_rxnorm_code: the ingredient rxnorm code for the prescription. Bar separated multi value field, where each value is an integer. Queried from RxMix combined with preexisting med rxnorm to ingredient rxnorm mappings. Approximate string matching is used when med rxnorm does not exist. Duplicate bar separated values for a row may exist. 2,125 unique values for rxnorm codes.
Right now the end date or duration of medication is not considered, see notes.

#### Note:

* I had difficulty calculate the relative amount of prescription each patient had. Pharmacist pointed at the total_med_quantity column in the original dataset.
It's not a surprise that it's the only messy column in the dataset, since everything else seems to be automatically generated. The value for total_med_quantity for even the same med_rxnorm can vary incredibly.
```
                            0              1             10        10 days
           270            106             16            988              5
       10 DAYS      10 TAB(S)        10 tabs           10.0            100
             3             20              1              4              4
            12             13             14        14 days      14 TAB(S)
             1             19           1324              2             75
          14.0             15      15 TAB(S)             16             17
             6              7              3             21              2
            18             19              2             20        20 days
            59            248              5          21771              1
      20 doses       20 pills      20 TAB(S)        20 tabs       20 tabs0
             1             11           2107            104              1
          20.0         20.000             21      22 TAB(S)             23
            22             18             25             11              2
            24             28      28 TAB(S)        28 tabs         28.000
            12           1360             92              4              4
            29            290              3             30      30 TAB(S)
            15              3              2            324              6
        30.000              4        4 weeks             40      40 TAB(S)
```

Without NLP this is imposslbe to deal with. I decided not to calculate the relative dosages.

* About grouping medicines by their functions, it's necessary to consider not only the ingredients, but the actual medicine, so rxnorm_code is a good idea. It's not certain. We have started with ingredients so I won't go back now.

* Right now I'm considering only prescription time. It's possible to use span, instead of the start and end points. Note that not all columns in this dataset has an end date, and that's the motivation to treat everything as a flag.

* Some missing med rx norms have been imputed if the med_name or med_generic exists in our database, this imputation is simlar to our lab loinc code imputation method

### Services
myserv.csv

87,086,734 rows

#### Columns:
* rep_person_id: 246743 unique.
* SRV_DATE: 1995-01-01 to 2016-12-31.
* srv_px_count: positive integer. should be a count for the services. However, there are many outliers in original table, such as negative values, so the interpretaion should be conservative. Tail chopped. Forced positive. default to 1.
* srv_px_code: string. HCP code or CPT code. **not** bar separated, because there are more than one values that need to be concatenated. I did not find a conversion method within my resources. Around 1:4 for HCP:CPT.
* SRV_LOCATION: string. 83 unique values after tail chopping. Coding scheme unknown.
* srv_admit_type: string, 19 unique. Ditto.
* srv_admit_src: string, 24 unique. Ditto.
* srv_disch_stat: strinag, 23 unique. Ditto.

#### Note:
* Tail chopped means all labels with total count less than threshold are merged to "other" label. Threshold is usually 1000.
* Admission and dispatch dates were thrown out

### Surgeries
mysurg.csv
2,275,007 rows

#### Columns:
* rep_person_id: 182,712 unique
* px_date: 1995-01-01 to 2016-12-31
* px_code: ICD-9 **procedure** codes, converted from gem_pcsi9.txt. 3353 unqiue.
* collapsed_px_code: collapsed twice for every code that has a count less than 1000, reduce sparsity. 907 unique. see prep.R for processing details

### Tobacco
This file contains a survey that is not processed into factors. Discarded completely.


### Vitals
myvitals.csv
33,167,683 rows

#### Columns:
* rep_person_id: 231,693 unique
* VITAL_DATE: 1995-01-01 to 2016-12-31
* BMI: see note, kg/m2
* BP DIASTOLIC: see note, mmHg
* BP SYSTOLIC: see note, mmHg
* HEIGHT: see note, cm
* WEIGHT: see note, kg

#### Note:
A person may have multiple vitals for one day, in that case, the average of the measurement is taken.
Units have been converted to metric, personal preferences.
I have been suggested to incorporate other statistics than the mean, such as sd. TODO 

## Data runtime processing with Python

Note that much of these are model-independent. I must keep these codes separately from the model.

### Loading
When we load a dataset#, we have a bunch of patients with different health records. The main difference is sparsity and length. Some patients visit often. Some patients stayed here for a short period.
This is very much what our original dataset looks like.
We will pull a patient number from the demographics and pull the patient's total medical records from other files.
At the moment I'm not convinced that making it a SQL database is going to improve performance. Since all files should be sorted, I need to exploit this property for fast access.
The loading of dataset needs to be cached and blazing fast. Everything needs to be loaded in memory.

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
