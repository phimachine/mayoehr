# okay, let's do it again.
# this time, we will adhere to a month of precision, because the date is easy to parse

require(lubridate)
require(data.table)
require(dplyr)
require(doParallel)
require(tidyr)

demo<-fread('/infodev1/rep/data/demographics.dat')
cod<-fread('/infodev1/rep/data/cause_of_death.csv')

main<-demo[,c("rep_person_id","birth_year","death_yes","death_date","sex","race","ethnicity","educ_level")]
mycod<-cod[,c("rep_person_id","death_date","age_years","sex","res_county","hospital_patient","underlying","injury_flag","code_type","code")]
# full join
main<-merge(mycod,main,all=TRUE)
# we do not want demographics information as targets, because doctors would know that.
# I also decided to throw out injury_flag, because only 5 records have 1, all others 0.
main<-main[,c("rep_person_id","death_date","underlying","code_type","code")]
# what is underlying?
# I will remove all HIC, for reason, see secondexplore.R, and I will convert all ICD9 to ICD10, one way or another.
# fixed: I will not throw out HIC entries, but I will remove the codes to be "other"
main<-main%>% mutate(code=if_else(code_type=="ICD10"|code_type=="ICD9",code,"")) %>% setDT()
# I want to convert all ICD9 to ICD10 


# I'm not going to expand 38 dimensions out of it. I think this should be done in python, as we convert to one-hot encoding. This is very straightforward process. Readlines until the rep_person_id/death_rate changes.
main<-main%>%arrange(rep_person_id)%>%setDT()
# we convert all the date time strings to date times. Time should not be important.
# run a few sample(mydia$dx_date,100), you will see that all are in the same format
main<-main%>% mutate(death_date=substr(death_date,1,9))%>%setDT()
main<-main%>% mutate(death_date=dmy(death_date))%>%setDT()
# this is good enough, we have enough for our target.

fwrite(main,"/infodev1/rep/projects/jason/deathtargets.csv")

##########
# FOR INPUTS, we will maintain separate files and read all by python at run time
# I think I will ditch the unique row by rep_person_id and dx_date convention. This means I will need to load every single dataset with python at run time, (and maybe pickle everything out?)
# this will add a O(1) when I convert to oe hot, but this should compensate for the file I/O
# since this dataset is presorted, the algorithm should be straightforward and optimal.

##### DIAGNOSIS
# diagnosis
dia<-fread('/infodev1/rep/data/diagnosis.csv')
# slicing necessary columns
mydia<-dia[,c("rep_person_id","dx_date","dx_codetype","dx_code_seq","DX_CODE")]
mydia<-mydia %>% filter(dx_codetype %in% c("I10","I9")) %>% setDT()
mydia<-mydia%>%mutate(dx_date=mdy(dx_date))%>%setDT()
# convert id to int before sort, otherwise it's string
hello<-as.integer(mydia$rep_person_id)
mydia<-mydia %>% mutate(rep_person_id=hello) %>% setDT()
mydia<-mydia %>% arrange(rep_person_id, dx_date) %>% setDT()
# merge demographics and mydia
fwrite(mydia,"/infodev1/rep/projects/jason/mydia.csv")

##### HOSPITALIZATION
hosp<-fread('/infodev1/rep/data/hospitalizations.dat')
myhosp<-hosp%>%select(rep_person_id, hosp_admit_dt,hosp_disch_dt,hosp_inout_code,hosp_adm_source,hosp_disch_disp, hosp_primary_dx,starts_with("hosp_secondary_dx"))
# no dirty data found, this is a carefully curated dataset.
# all of them seem to be ICD9 or ICD10
myhosp<-myhosp%>%mutate(hosp_admit_dt=mdy(hosp_admit_dt),hosp_disch_dt=mdy(hosp_disch_dt)) %>% setDT()
myhosp<-myhosp%>%arrange(rep_person_id,hosp_admit_dt)%>%setDT()
# myhosp has all the diagnosis codes expanded as dimensions
fwrite(myhosp,"/infodev1/rep/projects/jason/myhosp.csv")

##### LABS
labs<-fread("/infodev1/rep/data/labs.dat",fill=TRUE) 
mylabs<-labs%>%select(rep_person_id, lab_date, lab_src_code, lab_loinc_code, lab_result, lab_range, lab_units, lab_abn_flag)%>%setDT()
mylabs<-mylabs%>%mutate(lab_date=substr(lab_date,1,9))%>%setDT()
mylabs<-mylabs%>%mutate(lab_date=dmy(lab_date))%>%setDT()
mylabs<-mylabs%>%select(-lab_src_code)%>%setDT()
# I am going to normalize lab_range, lab_results and lab_units very naively. I believe this naive normalization method would work better than feeding it directly in.
# I assume that all lab results are normal distribution. I assumet that all lab_ranges are intervals based on sigmas.
# normalized_measure=(lab_result-up_or_lower_bound)/lab_range_interval_size
## this code stucks at one core
#lab_length<-nrow(mylabs)
#foreach(i=mylabs$lab_range,.combine='c')%dopar%{
#    strsplit(i,"-")
#}
## this code works, but somehow it's very slow.
#cl<-makeCluster(8)
#parLapply(cl,mylabs$lab_range,function(range){
#    strsplit(range,"-")
#})
# what's the point of 56 cores if you cannot use it?

## weird bug
#lab_length<-nrow(mylabs)
#splitted<-data.table(x=rep(0,lab_length),y=rep(0,lab_length))
#cl<-parallel::makeForkCluster(30)
#doParallel::registerDoParallel(cl)
#foreach(i=1:lab_length)%dopar%{
#    ss<-strsplit(mylabs[i],"-")    
##ss<-strsplit(mylabs[i,"lab_range"])
#    splitted[i,1]=ss[1]
#    if (!is.na(ss[2])){
#	splitted[i,2]=ss[2]
#    }
#}

# no parallel, 4 columns cover all.

splitted<- mylabs %>% separate(lab_range, c("X","Y"), sep="to", remove=FALSE) %>% setDT()
splitted<- splitted %>% separate(X,c("A","B","C","D"), sep='-', remove=FALSE) %>% setDT()
splitted<-splitted[,A:=as.double(A)][,B:=as.double(B)][,C:=as.double(C)][,D:=as.double(D)][,Y:=as.double(Y)]
# dirty cleaning:
# if A is empty and B has value, then we know the lab range start from negative, so we move B to A with negative, C to B, conditioned upon that there is no D value.
# condition holds that D has no value
sum(is.na(splitted$A) && !is.na(splitted$D))
# to mutate, we generate a new column and pass it, this avoids complicated function calls with dplyr
# we find those that need to converted
negative<-splitted[is.na(A)][!is.na(B)]
negative[,A:=-B]
negative[,B:=C]
negative[,C:=NA]
# e.g. -20 to 20
negative[!is.na(Y)][,"B"]<-negative[!is.na(Y)][,"Y"]
splitted[is.na(A)][!is.na(B)]<-negative
# some range are non standard. we will have to rely on the abnormality flag and assume it's +1
splitted[is.na(B)][!is.na(A)][,'A']<-NA
# now we have reached a point where extra information is clean.
# we want to see how abnormal a measurement is, then we will throw away all additional information and only retain the lab_delta and lab_abn_flag
splitted[,lab_result:=as.double(lab_result)]
# meaningful if smaller is negative
splitted_smaller<-splitted$lab_result-splitted$A
splitted_smaller<-replace_na(splitted_smaller,0)
splitted_smaller[splitted_smaller>0]<-0
splitted<-splitted %>% mutate(smaller=splitted_smaller) %>% setDT()
# meaningful if positive
splitted_bigger<-splitted$lab_result-splitted$B
splitted_bigger<-replace_na(splitted_bigger,0)
splitted_bigger[splitted_bigger<0]<-0
splitted<-splitted %>% mutate(bigger=splitted_bigger) %>% setDT()
# get range
splitted_range<-splitted$B - splitted$A
splitted_range<-replace_na(splitted_range,0)
# normalize
splitted<- splitted %>% mutate(range=splitted_range) %>% setDT()
splitted<- splitted %>% mutate(smaller=smaller/range, bigger=bigger/range) %>% setDT()
splitted[is.nan(smaller),'smaller']<-0
splitted[is.nan(bigger),'bigger']<-0
# we have finished normalization.
# for the last step, we can feed the abnormality flag to be a value, but I will not do it.
plitted<- splitted %>% select(rep_person_id,lab_date,lab_loinc_code,lab_abn_flag,smaller,bigger)
fwrite(splitted,"/infodev1/rep/projects/jason/mylabs.csv")

####### PRESCRIPTION
pres<-fread('/infodev1/rep/data/prescriptions.csv')
# I was not given a formula to precisely normalize the prescriptions.
# I have found that the med_route can be different for a med_rxnorm_code, but muchof the variations are free text, hard to analyze, and mostly mean the same, >60% are actually unique
mypres<-pres[med_rxnorm_code!=""]
# this condition filters out 40% of the rows. This is a big problem. Many of the med_generic/med_name does not have corresponding med_rxnorm_code and med_ingr_rxnorm_code.
# I can write an API in 3 days.

# take a look at those rows where sanity is TRUE
# this process needs to be run several times, but there might still be some missing
# this file is incredibly dirty
hello<-as.integer(mypres$rep_person_id)
mypres<-mypres[!is.na(hello)]
mypres<-mypres[rep_person_id!="0"]
mypres<-mypres %>% mutate(MED_DATE=dmy(substr(MED_DATE,1,9))) %>% setDT()
mypres<-mypres[!is.na(MED_DATE)]
# almost half of the data is gone at this point. I need to know where they are.
mypres<-mypres[nchar(med_rxnorm_code)<10]
mypres<-mypres[nchar(med_length_in_days)<6]
mypres<-mypres[nchar(med_self_reported)<3]
pres_table<-mypres %>% select(med_rxnorm_code) %>% group_by(med_rxnorm_code) %>% mutate (count=n()) %>% distinct(med_rxnorm_code, .keep_all=TRUE) %>% arrange(count) %>% setDT()
# I decided to throw out medications that are not used for more than 1000 times.

####### SERVICES
# my intuition tells me that sevices will not beo too vital
# I will filter our the tail of the dataset to control input complexity.
services_table <- serv %>% select(srv_px_code) %>% group_by(srv_px_code) %>% mutate(count=n()) %>% distinct(srv_px_code, .keep_all=TRUE) %>%  arrange(count) %>%  setDT()
services_table<- services_table[count>1000]
myserv<-serv %>% select (rep_person_id, SRV_DATE, srv_month, srv_px_code_type, srv_px_count, srv_px_code, SRV_LOCATION, srv_quantity, srv_age_years, SRV_ADT_DATE, srv_admit_type, srv_admit_src, SsRV_DISCH_DATE, srv_disch_stat)
myserv<- myserv[rep_person_id %in% services_table]
# I did not throw away any other dimensions' values. They are mainly noise
fwrite(mysev,"/infodev1/rep/projects/jason/myserv.csv")

######## SURGERIES
surg<-fread("/infodev1/rep/data/surgeries.dat")
mysurg<-surg%>%select(rep_person_id,px_date,px_codetype,px_code) %>%setDT()
# I am going to conver all I9 codes to ICD10 codes here. For table: https://www.cms.gov/medicare/coding/ICD10/2014-ICD-10-PCS.html
pcs<-fread("/home/m193194/git/ehr/death/data/gem_i9pcs.txt")
colnames(pcs)<-c("i9","i10","flag")
mysurg <- mysurg %>% separate(px_code,c("first","second"),remove=FALSE) %>% setDT()
# There is a warning message, but that's because our data base has codes that are very specific.
# discard because we have no lookup capability and to reduce dimension.
mysurg<-mysurg%>% unite("nodot",c("first","second"),sep="") %>% setDT()
# now we convert all I9 codes to I10 codes.
mysurg <- mysurg %>% mutate(nodot=if_else(px_codetype=="I9",nodot,"")) %>% mutate(nodot=as.integer(nodot)) %>% left_join(pcs, by=c("nodot"="i9")) %>% setDT()
# stop here, and you should see that all I9 has been converted.
mysurg <- mysurg %>% mutate(px_code=if_else(px_codetype=="I9",i10,px_code)) %>%  setDT()
mysurg <- mysurg %>% select (rep_person_id, px_date, px_code) %>% setDT()
# now we clean data again
# no error reported. this is a curated dataset.
mysurg <- mysurg %>% mutate(px_date=mdy(px_date)) %>% setDT()

# we have seen 45821 procedural codes, we need to collapse the dimensions.
# The algorithm is simple. We see the count for each 7 letter code, and if the count is fewer than 2000, then we aggregate them to a 5 letter code called "other"
# 2000 and 5 are arbitrary decisions
mysurg <- mysurg %>% group_by(px_code) %>% mutate(n=n()) %>% setDT()
mysurg <- mysurg %>% mutate(other=n<2000) %>% setDT()
# you can estimate how many will remain 7 letter code here. length(unique(count[other==FALSE]$px_code))
# 8923, reduced from 45821, not precise.
mysurg <- mysurg %>% mutate(collapsed_px_code=if_else(other==T,substr(px_code,1,5),px_code)) %>% setDT()
# not quite, we still have many codes, 5 letters is not enough. we repeat this process
mysurg <- mysurg %>% group_by(collapsed_px_code) %>% mutate(n=n()) %>% setDT()
mysurg <- mysurg %>% mutate(other=n<2000) %>% setDT()
mysurg <- mysurg %>% mutate(collapsed_px_code=if_else(other==T,substr(px_code,1,4),px_code)) %>% setDT()
# around 20000 is the sweet spot. You cannot compress more than this.
mysurg<-mysurg%>% select(-n,-other)
fwrite(mysurg,"/infodev1/rep/projects/jason/mysurg.csv")
# we need to rework. I will make a fork here. I should use ICD9 instead of ICD10 here.


######## 
