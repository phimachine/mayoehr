require('dplyr')
require('data.table')
require('reshape2')
require('ggplot2')

# get the dataset cause of death
death_cause<-fread('/infodev1/rep/data/cause_of_death.csv')
death_cause
summary(death_cause)
death_cause_length<-nrow(death_cause)
# we have 90294 data points
death_cause_length
# we have 16 column names
colnames(death_cause)
# gender is balanced, but the coding is not consistent
death_cause %>% group_by(sex) %>% summarize(n=n(),perc=n()/death_cause_length)
# ICD10 is the most common
death_cause %>% group_by(code_type) %>% summarise(n=n(),perc=n()/death_cause_length)
# many differnet disease code, but very compressed, 0.0287
length(unique(death_cause$code))
length(unique(death_cause$code))/death_cause_length
# summary by county. As you see, we have most people coming from Olmsted. Assuming that our application samples with the same demographic distribution, this would not be a problem. This could be a problem however, if we want to extrapolate our conclusion to a smaller county. We also see 4% missing data here. We also see inconsistent capitalization of text. I do not catch any misspelling by ascending order. 
death_cause %>% group_by(res_county) %>% summarize(n=n(),perc=n()/death_cause_length) %>% arrange(desc(perc))
# what are the data sources?
death_cause %>% group_by(data_source) %>% summarize(n=n())
# hospital_patient coding seems to be very inconsistent, largely (41%) missing
death_cause %>% group_by(hospital_patient) %>% summarize(n=n(),perc=n()/death_cause_length) %>% arrange(desc(perc))%>%print(n=Inf)
# I do not know what death unique id means, apparently there are more deaths than there are people.
length(unique(death_cause$death_internal_id))
length(unique(death_cause$rep_person_id))
rm(death_cause)

# crs_data_meds
# this is a big dataset, so we are going to get 10,000 points only for statistics
# by running head -1 crs_data_meds, we already know it's a sparse matrix, very inefficient coding.
crs_data_meds<-fread('/infodev1/rep/data/crs_data_meds.csv',nrows=10000)
# we see 3169 medications
colnames(crs_data_meds)
# how many medications do average patients take home? 11.6449
# a sparse vector where most people do not bring home any medication
summary(rowSums(crs_data_meds=="yes"))
# most of the medicines will not be prescribed at all
length(which(colSums(crs_data_meds=="yes")!=0))
rm(crs_data_meds)
# crs_data_presc is the same thing

# let's look at demographics
# we have 250,000 demographic data points
demo<-fread("/infodev1/rep/data/demographics.dat")
demo
summary(demo)
demo %>% group_by(sex) %>% count(n=n())
# everything I want to know is included in the summary. E.g. the education level is coded by numbers.
rm(demo)

# let's look at diagnosis, first 10000
dia<-fread("/infodev1/rep/data/diagnosis.csv",nrows=10000)
dia
# seems to contain the background for each diagnosis.
# the vital information is contained in DX_CODE and dx_descr
# dx_descr is clinical notes
# most of the dx_code_type is I9, a numerical coding, e.g. 272.4, v04.81
dia%>%group_by(dx_codetype)%>%summarize(n=n())     
rm(dia)

# hospitalization
# it contains the admission datetime and place.
# some codes provide more info.
hosp<-fread("/infodev1/rep/data/hospitalizations.dat")
# if you take the head and tail, you will see that a lot of data is missing,
# mostly the codes, e.g. I34.1. Looks like I9
summary(hosp=="")
# it should be clear that each note belongs to a category. Most of the time
# it's empty. Some are very sparse. (1:100 missing for dx_23)
rm(hosp)


# labs
labs<-fread("/infodev1/rep/data/labs.dat",nrows=10000)
# labs are very good data. It's low dimensional, unexpectedly. 
labs
# for example, lab_loinc_desc is in natural text, but I would assume
# that it's redundant with the lab_loinc_code
# lab_units may not be consistent. I should run an experiment to see if
# neural network needs help normalizing those units.
# despite the fact that lab.dat is small, the neural network is to be 
# incredibly big, since different tests are very different and cannot 
# share the same module.
# dates are not parsed, so I cannot run summary.
# lab_abn_flag is useful.
rm(labs)

# prescription
pres<-fread("/infodev1/rep/data/prescriptions.csv",nrows=10000)
# prescription is not a sparse big matrix. I wonder how this is different from the crs_data_pres. This seems to contain more information, with notes and
# dates.
pres
colnames(pres)
# med instructions, for example are human readble terminologies with a lot of acronyms. They should be mapped to a feature space first.
pres$med_instructions[1:100]
# notes are very sparse, to a point you can throw it out to reduce model complexity
summary(pres$med_notes=="")

# rep_data.csv
# by running head rep_data.csv, we know it contains a list of clinical numbers

# rep_data_demo.csv
# by running head reP_data_demo.csv, we see that
# this seems to be a file that joined demographics file and rep_data.csv

# Talking to David, I know that all rep_data_xxx.csv is processed from raw data.
# What is the idea behind the processing?
# rep_data_labs.csv, rep_data_meds.csv and rep_data_surg.csv are sparse one-hot encoding
# both throw away information such as date time and locations
# they treat all columns to be parallel, ignore the correlation among medicines
# 9correlation among medicines are not hard to achieve, we can observe the prescription context and throw them on a feature space.
# processing feature vectors than the one-hot encoding is much more efficient.
# this can be done with end-to-end.

# other files such as rep_data_hosp.csv compressed the information, counted
# the number of admissions and threw away date/time, diagnosis code

# all of my conclusions are drawn by probing head rep_data_xxx.csv
# we will need to process data in a way that exploits the advantage of neural networks. Much of this information should be retained.

# services.dat
serv<-fread("/infodev1/rep/data/services.dat",nrows=10000)
serv
serv[200:210]
# some nursing home information, admission time and location. not very useful.
colnames(serv)
# there is no notes.
rm(serv)
# surgeries.dat

# surgeries.dat
surg<-fread("/infodev1/rep/data/surgeries.dat", nrow=10000)
surg
# I9 px_code are very useful.
# how many surgeries don't have px_code? None.
sum(surg$px_code=="")
sum(surg$px_code=="NA")
surg$px_code
# px_descr is not patient-specific. It's just px_code annotations
# this is useful to aid feature interpretation of px_code
surg %>% group_by(px_code,px_descr) %>% summarize() %>% print(n=Inf)
# there are some columns like px_plusfour that does not make much sense to me
rm(surg)


# tobacco_status.dat
# reading this file for first 10000 lines cause crash
toba<-fread("/infodev1/rep/data/tobacco_status.dat")
toba
# tobacco seems to be a survey file. The questions and answers are largly qualitative. Pretty trivial and difficult to analyze.

# vitals.dat
vitals<-fread("/infodev1/rep/data/vitals.dat",nrows=10000)
vitals[2000:2100]
# it's not sparse at all. problem is that each row can be very different vital types
sum(vitals$VITAL_SRC_DESC=="")


# INVESTIGATION

# I want to know how many diagnoses, prescriptions and surgeries people have on average
# this is an estimate
dia<-fread("/infodev1/rep/data/diagnosis.csv",nrows=10000)
cn<-fread("/infodev1/rep/data/rep_data.csv")
unique_list<-unique(dia$rep_person_id)
unique_list %in% cn
# none of the diagnosis clinical number exists in the rep_data.csv
# let's see about the surgeries.dat
surg<-fread("/infodev1/rep/data/surgeries.dat",nrows=10000)
# whatis rep_data.csv? it does not contain anything in the raw file
# it's useful to have a list of all clinical numbers in our file, but it does not actually exist yet.
sum(unique_list %in% cn)

# plot a bar plot by year, I expect most data to be from recent years
# for this, I will need to read the whole dataset, becuase I'm afraid the file is ordered by time
# diagnosis
if (FALSE){
'''
 dia<-fread("/infodev1/rep/data/diagnosis.dat")
|--------------------------------------------------|
|==================================================|
|--------------------------------------------------|
|==================================================|
Warning message:
In fread("/infodev1/rep/data/diagnosis.dat") :
  Stopped early on line 2710113. Expected 19 fields but found 10. Consider fill=TRUE and comment.char=. First discarded non-empty line: <<505028|12/27/1997|12|27|1997|REP Diagnostic Index|HIC|1|07781610|SYNDROME, RESPIRATORY DISTRESS, NOS--RDS (ADULT)>>
'''

> dia<-fread("/infodev1/rep/data/diagnosis.csv")
|--------------------------------------------------|
|==================================================|
|--------------------------------------------------|
|==================================================|
> dim(dia)
[1] 102305026        19
}

# what year is 0001? 0005?
# it does not make any sense
ytb<-table(dia$dx_year)
if (FALSE){
              0    0001    0005    0014    0068    0069    0082    0095    0101
  31122       1       1       1       1       1       2       3       1       2
   0102    0107    0125    0150    0151    0195    0207    0212    0217    0243
      2       1       1       1       1       1       1       2       1       2
   0248    0272    0277    0286    0325    0326    0352    0353    0357    0363
      1       1       1       1       1       1       1       3       2       2
   0367    0369    0371    0379    0401    0407    0425    0427    0434    0443
      1       3       1       1       1       1       1       1       2       1
   0444    0447    0450    0456    0457    0466    0467    0481    0492    0508
      1       1       1       2       1       4       1       2       1       3
   0520    0543    0545    0546    0557    0564    0565    0582    0585    0586
      1       2       1       1       1       1       1       1       1       3
   0587    0605    0607    0610    0634    0647    0651    0653    0674    0695
      1       1       1       3       1       1       2       1       1       1
   0708    0713    0719    0742    0779    0782    0800    0808    0813    0819
      1       1       2       3       1       1       5       1       1       1
   0825    0842    0847    0870    0871    0901    0903    0904    0908    0909
      1       1       1       1       1       2       1       1       1       1
   0927    0928    0934    0950    0956    0959    0961    0962    0964    0976
      1       1       2       1       3       1       1       1       2       1
    100    1000    1001    1003    1006    1007    1009     101    1010    1011
      1       2       3       1       2       1       1       1       1       3
   1014    1015    1018    1020    1021    1022    1023    1024    1025    1026
      4       3       1       2       1       1       2       1       1       2
   1027    1029    1030    1032    1034    1038     104    1040    1041    1045
.....
}
# we can see a general distribution of the dataset
yytb<-data.table(ytb)
yytb[1993<V1 & V1<2018]
# there is a weird number, I don't get it
yytb<-yytb[1993<V1 & V1<2018][-6]
# some graph
qplot(yytb$V1,yytb$N)
ggsave("/home/m193194/Desktop/plot1.png")
# go see the graph, you will see that it's not exponential.
# around 2001 it's at the top already.
# recall, this is the diagnosis file.
# we do the same for prescription, for example
surg<-fread("/infodev1/rep/data/surgeries.dat")
stb<-table(surg$px_year)
# clean and beautiful
stb
stb<-data.table(stb)
qplot(stb$V1,stb$N)
ggsave("/home/m193194/Desktop/plot2.png")
# follows the same general trend, but after 2007, the number of surgeries
# is quartered, compared to 25% decrease for diagnosis, not sure the reason
# does this require normalization? I don't think so.
# the magnitude is different, of course.

# plot a bar plot by month, I expect seasonal change
# let's see diagnosis first
dtb<-data.table(table(dia$dx_month))
dtb<-dtb[,V1:=as.integer(V1)]
dtb<-dtb[V1<13]
qplot(dtb$V1,dtb$N)
ggsave("/home/m193194/Desktop/plot3.png")
# let's see surgeries
stb<-data.table(table(surg$px_month))
qplot(stb$V1,stb$N)
ggsave("/home/m193194/Desktop/plot4.png")
# no apparent pattern. Not noise either, since the sample size is pretty big.

# do a sanity check on some demographics ratio of some dataset
# do male have more surgeries than female?
# first, we check if we can use the demographics.csv file for surgeries.dat
>sum(surg[1:100000]$rep_person_id %in% demo$rep_person_id)
[1] 100000
# good news.
# then, lets join surgery and demographics sex
surg_people<-surg[,c("rep_person_id")]
demo_sex<-demo[,c("rep_person_id","sex")]
mm<-merge(unique(surg_people),demo_sex)
table(mm$sex)
# 94452 F and 88215 M
# what about race? This is the baseline
table(demo$race)

     1      2      3      4      5      6     98     99
 14935  11825    552    972  13421 186684   1201  20465

demo_race<-demo[,c("rep_person_id","race")]
mm<-merge(unique(surg_people),demo_race)
table(mm$race)
     1      2      3      4      5      6     98     99
  9653   7346    401    645   8847 148185    578   7057

# this is probably easier to look at
# those who id not have surgeries
> prop.table(table(demo$race))
          1           2           3           4           5           6
0.059726860 0.047289596 0.002207514 0.003887145 0.053672192 0.746571754
         98          99
0.004802943 0.081841995

# those who had surgeries
> prop.table(table(mm$race))
          1           2           3           4           5           6
0.052831779 0.040205350 0.002194711 0.003530146 0.048420465 0.811030474
         98          99
0.003163448 0.038623626
# might reflect difference in health care coverages among races, race 6 
# seems to be the wealthiest and the only one increasing with surgery percentage.


# I want to know how many diagnosis there are compared to all demographics records.
# there are 247248 people in diagnosis, compared to 250,056 people in demographics
length(unique(dia$rep_person_id))
# and we have demographics information for all but 2.
sum(unique(dia$rep_person_id) %in% demo$rep_person_id
# we have 18562 people died, that is 7% of demo
length(unique(deaths$rep_person_id))
# all of them are in demo
sum(unique(deaths$rep_person_id) %in% demo$rep_person_id)
# we have 192049 people in hospitalizaiton, that's a lot, all of them in demo
sum(unique(hosp$rep_person_id)%in% demo$rep_person_id)
# we have 182712 people with surgeries. I don't get it. All rows are non-empty
length(unique(surg$rep_person_id))
# we have all of them in demo
sum(unique(surg$rep_person_id) %in% demo$rep_person_id)

# average entry frequencies for each patient is 12. Not too good. Not impossible.
row(surg)/182712

# I want to know the time span of each record
try<-surg%>%group_by(rep_person_id)%>% mutate(span=max(px_year)-min(px_year),max=max(px_year),min=min(px_year),count=n())%>% select(rep_person_id,span,max,min,count)
try<-data.table(try)
ditin<-distinct(try,rep_person_id,span,max,min,count)
# result
        rep_person_id span  max  min count
     1:        110925   20 2015 1995    28
     2:          7504    1 1996 1995     4
     3:         51783   13 2010 1997     3
     4:         75202   20 2015 1995    75
     5:        100414   10 2006 1996     4
    ---
182708:        115657    0 2005 2005     2
182709:        546495    0 2016 2016     9
182710:        776547    0 2016 2016    11
182711:       1027389    0 2013 2013     2
182712:        475385    0 2016 2016     1
# the highest visit is 490
> ditin[order(ditin$count)]
        rep_person_id span  max  min count
     1:         92885    0 2001 2001     1
     2:         39308    0 1996 1996     1
     3:        157277    0 1998 1998     1
     4:        110572    0 1995 1995     1
     5:         13663    0 1998 1998     1
    ---
182708:        741641   15 2016 2001   358
182709:        360182   15 2012 1997   400
182710:        244531   14 2009 1995   427
182711:        332527   16 2011 1995   465
182712:         34008   17 2012 1995   490

# one-hot feasibility?
> length(unique(dia$DX_CODE))
[1] 57762
> nrow(dia)
[1] 102305026

# can our network handle an input dimension of 10e+5?
# our story input, even batched, are only 10e+3.
# memory. Memory. MEMORY!
