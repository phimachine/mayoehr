# okay, let's do it again.
# this time, we will adhere to a month of precision, because the date is easy to parse

require(lubridate)
require(data.table)
require(dplyr)
require(doParallel)
require(foreach)
require(parallel)
require(tidyr)
require(fuzzyjoin)
require(xml2)
require(XML)
require(stringr)

########## DEATH TARGETS
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
# we will covert HICDA to ICD9, throw away BRK
hic<-fread('/home/m193194/git/ehr/death/data/hicda_icd9.csv')
colnames(hic)<-c("hicda","hicda_desc","icd9","icd9_desc","grpnbr")
main <- main %>% filter(code_type!="BRK") %>% setDT()
# transform the hic conversion table for our application..
hello <- hic %>% distinct(hicda, .keep_all=T) %>% setDT()
hello <- hello %>% select(hicda,icd9) %>% setDT()
hello <- hello %>% mutate(hicda=as.character(hicda)) %>% setDT()
hello<- hello %>% mutate(hicda=str_pad(hicda,8,pad="0")) %>% setDT()
main <- main%>% left_join(hello,by=c("code"="hicda")) %>% setDT()
# merge back HIC converted
main <- main %>% mutate(code=if_else(code_type=='HIC',icd9,code)) %>% select(-icd9) %>%  setDT()
# I want to convert all ICD10 to icd9
gem<-fread('/home/m193194/git/ehr/death/data/2018_I10gem.txt')
colnames(gem)<-c("i10","i9","flags")
# this gem file has missing rows, and it's going to be problematic. We need to manually generate more conversions by guessing it.
# we probably don't need to do it in python. We have fuzzyjoin library.
require(fuzzyjoin)
# no dot, because our conversion table
main <- main%>% separate(code,c("first","second"),remove=FALSE) %>%  setDT()
main <- main %>% replace_na(list(first="",second="")) %>% setDT()
main <- main%>% unite("nodot",c("first","second"),sep="") %>% setDT()
main <- main%>% mutate(nodot=if_else(code_type=="ICD10",nodot,"")) %>% setDT()
# potentially possible for parallel computation, this takes forever
start <- proc.time() # Start clock
regex_result <- gem %>% regex_right_join(main,by=c(i10="nodot")) %>% setDT()
time_elapsed_singlecore <- proc.time() - start # End clock###   
# This code will not work because our workstation does not have libssl.
#start<- proc.time()
#library(parallel)
#library(multidplyr)
#cl<- detectCores()-4
#main <- main %>% mutate(group_index=rep(1:cl, length.out=nrow(main))) %>% setDT()
#cluster<-create_cluster(cores=cl)
#by_group<-main %>% partition(group_index, cluster=cluster)
#by_group %>%
#	cluster_library("fuzzy<) %>%
#	cluster_library("data.table") %>%
#	cluster_library("dplyr") %>%
#	cluster_assign_value("gem",gem)
#regex_result_parallel<-by_group %>% gex_right_join(main,by=c(i10="nodot")) %>% collect() %>% setDT()
#time_elapsed_parallel<- proc.time()-start
#
# process regex_result
regex_result <- regex_result %>% distinct(rep_person_id,code,underlying,.keep_all=T) %>% setDT()
main <- regex_result %>% mutate(code=if_else(code_type=="ICD10",i9,code)) %>% setDT()
# HIC still has dots. remove
main <- main%>% separate(code,c("first","second"),remove=FALSE) %>%  setDT()
main <- main %>% replace_na(list(first="",second="")) %>% setDT()
main <- main%>% unite("nodot",c("first","second"),sep="") %>% setDT()
main <- main %>% mutate(code=if_else(code_type=="HIC"|code_type=="ICD9",nodot,code)) %>% select(rep_person_id,death_date, underlying,code_type,code) %>% setDT()

# after collapsing, we will have fewer rows than before, because the cause of death dataset has records spanning more than one line.
#####hello88 <- main %>% group_by(rep_person_id, code, underlying) %>% mutate(n=n()) %>% setDT()
#####hello88 <- hello88 %>% arrange(rep_person_id, code) %>% setDT()
#####hello88[n!=1]

# underlying is redundant
under <- main %>% select(rep_person_id, code, underlying) %>% setDT()
under <- under %>% group_by(rep_person_id, code ) %>% mutate(n=n())
under <- under %>% mutate(underlying=n==2) %>% setDT()
under <- under %>% distinct(rep_person_id,code,underlying) %>% setDT()
main <- main %>% distinct(rep_person_id, code, .keep_all=T) %>% setDT()
main <- main %>% mutate(underlying=under$underlying) %>% setDT()

# I'm not going to expand 38 dimensions out of it. I think this should be done in python, as we convert to one-hot encoding. This is very straightforward process. Readlines until the rep_person_id/death_rate changes.
main<-main%>%arrange(rep_person_id)%>%setDT()
# we convert all the date time strings to date times. Time should not be important.
# run a few sample(mydia$dx_date,100), you will see that all are in the same format
main<-main%>% mutate(death_date=substr(death_date,1,9))%>%setDT()
main<-main%>% mutate(death_date=dmy(death_date))%>%setDT()
# around 7000 points have no corresponding ICD9. I manually examined a few, and I think they are mostly data entry errors. Most of ICD10 codes don't exist on ICD10data.com
main <- main %>% filter(!is.na(code))%>% select(-code_type) %>% setDT()
# we have 59394 rows in the end
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
mydia<-mydia%>%mutate(dx_date=mdy(dx_date))%>%setDT()
# convert id to int before sort, otherwise it's string
hello<-as.integer(mydia$rep_person_id)
mydia<-mydia %>% mutate(rep_person_id=hello) %>% setDT()
mydia<-mydia %>% arrange(rep_person_id, dx_date) %>% setDT()
# 50,000/100,000,000 out
mydia<- mydia[!is.na(rep_person_id)][!is.na(dx_date)]
# rid of hicda
hic<-fread('/home/m193194/git/ehr/death/data/hicda_icd9.csv')
colnames(hic)<-c("hicda","hicda_desc","icd9","icd9_desc","grpnbr")
# transform the hic conversion table for our application..
hello <- hic %>% distinct(hicda, .keep_all=T) %>% setDT()
hello <- hello %>% select(hicda,icd9) %>% setDT()
hello <- hello %>% mutate(hicda=as.character(hicda)) %>% setDT()
hello<- hello %>% mutate(hicda=str_pad(hicda,8,pad="0")) %>% setDT()
mydia<- mydia %>% left_join(hello,by=c("DX_CODE"="hicda")) %>% setDT()
mydia<- mydia %>% mutate(DX_CODE=if_else(dx_codetype=="HIC", icd9, DX_CODE)) %>% select(-icd9) %>% setDT()
mydia<- mydia %>% mutate(dx_codetype=if_else(dx_codetype=="HIC","I9",dx_codetype)) %>% setDT()
# transform I10 to I9 TODO
pcs<-fread('/home/m193194/git/ehr/death/data/2018_I10gem.txt')
colnames(pcs)<-c("i10","i9","flag")
pcs<- pcs %>% distinct(i10,.keep_all=T) %>% select(-flag) %>% setDT()

mydia<- mydia %>% separate(DX_CODE,c("first","second"), remove=F)
mydia <- mydia %>% replace_na(list(first="",second="")) %>% setDT()
mydia <- mydia %>% unite("nodot", c("first","second"),sep="") %>% setDT()

# do not modify
ret<-fread("/infodev1/rep/projects/jason/mydiag_rett.csv")
# for the regex join, we will need to run it efficiently.
all_i10_codes <- mydia %>% filter(dx_codetype=="I10") %>% distinct(nodot) %>% setDT()
library(foreach)
library(parallel)
# parallel processing needs to be done here. Each lookup costs around 3 seconds. We have ~23,000 in total.
cl<-parallel::makeForkCluster(32)
doParallel::registerDoParallel(cl)
ret<-foreach(i=1:nrow(all_i10_codes), .combine='rbind') %dopar%{
    pcs%>% regex_right_join(all_i10_codes[i],by=c(i10="nodot"))

stopCluster(cl)
# Throws error if 48 clusters
# Error in unserialize(socklist[[n]]) : error reading from connection
fwrite(ret,"/infodev1/rep/projects/jason/mydia_reg")
ret<-ret%>% select(-i10) %>% setDT()

# this is going to produce 1 billion rows the end
# this is because we do matches on i9 codes too, and they have mutliple matches. This piece of code is very bad. We could've run the regex match then leftjoin
mydia<- mydia %>% left_join(ret,by=c("nodot"="nodot")) %>% setDT()
mydia <- mydia %>% mutate(DX_CODE=if_else(dx_codetype=="I10",i9, nodot)) %>% setDT()
mydia <- mydia %>% select(-dx_codetype,-nodot,-i9,-dx_code_seq) %>% setDT()
mydia <- mydia %>% distinct(rep_person_id,dx_date,DX_CODE) %>% setDT()

fwrite(mydia,"/infodev1/rep/projects/jason/mydia_before_bar")
fwrite(as.list(unique(mydia$DX_CODE)),"/infodev1/rep/projects/jason/mydia_all_dx_codes")
# I like bar separated notations
mydia[DX_CODE==""]$DX_CODE<-"empty"
mydia <- mydia %>% group_by(rep_person_id,dx_date) %>% mutate(dx_codes=paste0(DX_CODE,collapse="|")) %>% setDT()
mydia <- mydia %>% distinct(rep_person_id,dx_date,.keep_all=T) %>% setDT()
mydia <- mydia%>% mutate(dx_date=mdy(dx_date)) %>% setDT()
fwrite(mydia,"/infodev1/rep/projects/jason/mydia_no_distinct.csv")
mydia <- mydia %>% distinct(rep_person_id,dx_date,dx_codes) %>% setDT()
mydia <- mydia %>% arrange(rep_person_id, dx_date) %>% setDT()
fwrite(mydia,"/infodev1/rep/projects/jason/mydia.csv")

#########################
##### HOSPITALIZATION
hosp<-fread('/infodev1/rep/data/hospitalizations.dat')
myhosp<-hosp%>%select(rep_person_id, hosp_admit_dt,hosp_disch_dt,hosp_inout_code,hosp_adm_source,hosp_disch_disp, hosp_primary_dx,hosp_dx_code_type,starts_with("hosp_secondary_dx"))
# no dirty data found, this is a carefully curated dataset.
# all of them seem to be ICD9 or ICD10
myhosp<-myhosp%>%mutate(hosp_admit_dt=mdy(hosp_admit_dt),hosp_disch_dt=mdy(hosp_disch_dt)) %>% setDT()
myhosp<-myhosp %>% arrange(rep_person_id,hosp_admit_dt)%>%setDT()
# myhosp has all the diagnosis codes expanded as dimensions

# covert all expanded column TODO
myhospcopy<-copy(myhosp)
myhosp<-copy(myhospcopy)

pcs<-fread('/home/m193194/git/ehr/death/data/2018_I10gem.txt')
colnames(pcs)<-c("i10","i9","flag")
pcs<- pcs %>% distinct(i10,.keep_all=T) %>% select(-flag) %>% setDT()

vi10<-c()
# collect all unique i10, so that regex join does not go through all rows.
##### THIS FUNCTION COLLECTS NOT ONLY I10 BUT ALSO I9, so we cater to it later
collect_i10<- function(myhosp,key,vi10){
    myhosp<- myhosp%>% separate(key,c("first","second"), remove=F)
    myhosp <- myhosp %>% replace_na(list(first="",second="")) %>% setDT()
    myhosp <- myhosp %>% unite("nodot", c("first","second"),sep="") %>% setDT()
    vi10<-c(vi10,myhosp$nodot)
    vi10
}
# select all expanded columns
keys<- myhosp[1:2] %>% select(starts_with("hosp_secondary_dx")) %>% colnames()
keys<-c("hosp_primary_dx",keys)
# collect all i10 codes
for (key in keys) {
    print(key)
    vi10<-collect_i10(myhosp,key,vi10)
}
# convert all i10 codes. Two passes are more efficient
uvi10<-unique(vi10)
uvidt<-data.table(i10=uvi10)
uvidt<-uvidt[i(i10=="")]
# should take around 1 hour with this configuration
cl<-parallel::makeForkCluster(48)
doParallel::registerDoParallel(cl)
ret<-foreach(i=1:nrow(uvidt),.combine='rbind') %dopar%{
    pcs %>% regex_right_join(uvidt[i],by=c(i10="i10"))
}
stopCluster(cl)
# so I remember that the list contains i9 codes too
colnames(ret)<-c("pcs_match","i9", "uvi10i9")
ret <- ret%>% distinct(uvi10i9,.keep_all=T) %>% setDT()
fwrite(ret,"/infodev1/rep/projects/jason/myhosp_reg")
ret <- ret %>% select(-pcs_match) %>% setDT()
# replace all i10 with the corresponding regex result.
# normal left join
batch_to_i9 <- function(myhosp,key,regret){
    myhosp <- myhosp %>% separate(key,c("first","second"), remove=F)
    myhosp <- myhosp %>% replace_na(list(first="",second="")) %>% setDT()
    myhosp <- myhosp %>% unite("nodot", c("first","second"),sep="") %>% setDT()
    myhosp <- myhosp %>% left_join(regret, by=c("nodot"="uvi10i9")) %>% setDT()
    # i9 codes have nonsensical matches, but they will not be used here
    myhosp <- myhosp %>% mutate(!!key:=if_else(hosp_dx_code_type=="I10", i9, nodot)) %>% setDT()
    myhosp <- myhosp %>% select(-i9) %>% setDT()
    myhosp
}


# combine back
for (key in keys){
    print(key)
    myhosp<-batch_to_i9(myhosp,key,ret)
}

# MANUALLY EXAMINE
hosp<- hosp %>% arrange(rep_person_id, hosp_admit_dt) %>% setDT()
myhosp<- myhosp %>% arrange(rep_person_id, hosp_admit_dt) %>% setDT()
myhosp<- myhosp %>% select(-nodot) %>% setDT()

# MERGE COLUMNS
# merge with space
for (key in keys) set(myhosp,which(is.na(myhosp[[key]])),key,"")
fwrite(myhosp,"/infodev1/rep/projects/jason/myhosp_before_merge.csv")

myhosp<- myhosp %>% unite("all_dx_codes",keys, sep=" ") %>% setDT()
splitted<-strsplit(myhosp$all_dx_codes,"\\s+")
# split with space, replace with bar
pasted<-lapply(splitted,function(x) paste(x,collapse='|'))
# mutate
myhosp<- myhosp %>% mutate(dx_codes=pasted) %>% setDT()
myhosp<- myhosp %>% select(-all_dx_codes,-hosp_dx_code_type) %>% setDT()

# for the two labels, remove all below 1000 cases.
myhosp<-myhosp %>% group_by(hosp_disch_disp) %>% mutate(n=n()) %>% setDT()
myhosp<-myhosp %>% mutate(hosp_disch_disp=if_else(n<1000,"XXX",hosp_disch_disp)) %>% select(-n) %>% setDT()

myhosp<-myhosp %>% group_by(hosp_adm_source) %>% mutate(n=n()) %>% setDT()
myhosp<-myhosp %>% mutate(hosp_adm_source=if_else(n<1000,"XXX",hosp_adm_source)) %>% select(-n) %>% setDT()

myhosp<-myhosp %>% mutate(is_in_patient=hosp_inout_code=="I") %>% select(-hosp_inout_code) %>% setDT()
myhosp<-myhosp %>% arrange(rep_person_id, hosp_admit_dt) %>% setDT()
myhosp<- myhosp %>% mutate(dx_codes=unlist(dx_codes)) %>% setDT()
myhosp[hosp_disch_dt=="1800-01-01"]$hosp_disch_dt<-NA
fwrite(myhosp,"/infodev1/rep/projects/jason/myhosp.csv")

##### LABS
labs<-fread("/infodev1/rep/data/labs.dat",fill=TRUE) 
mylabs<-labs%>%select(rep_person_id, lab_date, lab_src_code, lab_loinc_code, lab_result, lab_range, lab_units, lab_abn_flag)%>%setDT()
mylabs<-mylabs%>%mutate(lab_date=substr(lab_date,1,9))%>%setDT()
mylabs<-mylabs%>%mutate(lab_date=dmy(lab_date))%>%setDT()

# impute missing lab_loinc_code given lab_src_code
failsafe<- mylabs[,c("lab_loinc_code","lab_src_code")]
failsafe<- failsafe %>% group_by(lab_loinc_code,lab_src_code) %>% mutate (n=n()) %>% setDT()
failsafe<- failsafe %>% group_by(lab_loinc_code) %>% mutate(maxn=max(n)) %>% setDT()
failsafe <- failsafe %>% filter(n==maxn) %>% select(-n) %>% setDT()
failsafe<- failsafe %>% distinct(lab_src_code,.keep_all=T) %>% setDT()
failsafe<- failsafe %>% select(-maxn) %>% setDT()
# there is only one, count too big that I cannot remove it.
failsafe[lab_loinc_code==""]$lab_loinc_code<-"2099-ROCLIS"

failsafe<- failsafe %>% mutate(new_loinc=lab_loinc_code) %>% select(-lab_loinc_code) %>% setDT()
mylabs<- mylabs %>% left_join(failsafe) %>% setDT()
mylabs<- mylabs %>% mutate(lab_loinc_code=if_else(lab_loinc_code=="", new_loinc, lab_loinc_code)) %>% setDT()
mylabs<- mylabs %>% select(-new_loinc) %>% setDT()

# we still need to clean up and impute the missing values
mylabs<-mylabs %>% mutate(rep_person_id=as.integer(rep_person_id)) %>% setDT()
mylabs <- mylabs[!is.na(rep_person_id)][rep_person_id!=0][!is.na(lab_date)][lab_loinc_code!=""][!is.na(lab_loinc_code)]




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
# we can feed the abnormality flag to be a value, but I will not do it.

# we still need to clean up and impute the missing values
splitted<- splitted %>% select(rep_person_id,lab_date,lab_loinc_code,lab_abn_flag,smaller,bigger)
splitted<- splitted %>% arrange(rep_person_id, lab_date) %>% setDT()
splitted<-coll(splitted,"lab_abn_flag",1000,"Unknown")
fwrite(as.list(unique(splitted$lab_loinc_code)),"/infodev1/rep/projects/jason/all_lab_loinc_codes.csv")
fwrite(splitted,"/infodev1/rep/projects/jason/mylabs.csv")

####### PRESCRIPTION
pres<-fread('/infodev1/rep/data/prescriptions.csv')
# I was not given a formula to precisely normalize the prescriptions.
# I have found that the med_route can be different for a med_rxnorm_code, but muchof the variations are free text, hard to analyze, and mostly mean the same, >60% are actually unique

# this file is incredibly dirty, we clean up here before doing anything else.
# it turns out that this file is very bad, so we are going to do it column by column rigorously.
mypres<-copy(pres)
mypres<-mypres%>% mutate(rep_person_id=as.integer(rep_person_id)) %>% filter(!is.na(rep_person_id)) %>% setDT()
mypres<-mypres[rep_person_id!="0"]
mypres<-mypres %>% mutate(MED_DATE=dmy(substr(MED_DATE,1,9))) %>% setDT()
# most of them are I/O errors here. very few rows. must be discarded for time series.
# string processing is always very slow. We might want to consider parallel processing here.
mypres<-mypres[!is.na(MED_DATE)]
mypres<-mypres[nchar(med_name)<100]
mypres<-mypres[nchar(med_generic)<100]
mypres<-mypres[nchar(med_strength)<100]
mypres<-mypres[nchar(med_form)<100]
mypres<-mypres[nchar(med_route)<100]
mypres<-mypres[nchar(med_dose)<100]
mypres<-mypres[nchar(med_dose_units)<100]
mypres<-mypres[nchar(med_frequency)<100]
# 11980544 rows
mypres<-mypres[nchar(med_duration)<100]
mypres<-mypres[nchar(med_total_quantity)<100]
mypres<-mypres[nchar(med_refills)<100]
mypres<-mypres[nchar(med_instructions)<100]
mypres<-mypres[nchar(med_indication)<100]
#mypres<-mypres %>% mutate(med_update_date=dmy(substr(med_update_date,1,9))) %>% setDT()
mypres<-mypres[nchar(med_notes)<1000]
mypres<-mypres[nchar(med_rxnorm_code)<100]
mypres<-mypres[nchar(med_rxnorm_desc)<400]


mypres<-mypres[nchar(med_ndfrt_class)<100]
mypres<-mypres[nchar(med_ndfrt_class_desc)<100]
mypres<-mypres[nchar(med_ndfrt_header)<100]
mypres<-mypres[nchar(med_ndfrt_header_desc)<400]
mypres<-mypres[nchar(med_ingr_rxnorm_code)<100]
mypres<-mypres[nchar(med_ingr_rxnorm_desc)<200]
mypres<-mypres[nchar(med_self_reported)<10]
mypres<-mypres[nchar(med_length_in_days)<10]
mypres<-mypres[nchar(med_end_date)<30]
mypres<-mypres[nchar(med_src)<30]
# 116228058
# test med_name dirty data
# test<- mypres %>% group_by(med_name) %>% mutate(n=n()) %>% distinct(med_name, n) %>% arrange(n) %>%  setDT()
# test<-test[n==1]
# test[sample(nrow(test),10)]

# pres_table<-mypres %>% select(med_rxnorm_code) %>% group_by(med_rxnorm_code) %>% mutate (count=n()) %>% distinct(med_rxnorm_code, .keep_all=TRUE) %>% arrange(count) %>% setDT()
# mypres<-pres[med_rxnorm_code!=""]
# this condition filters out 40% of the rows. This is a big problem. Many of the med_generic/med_name does not have corresponding med_rxnorm_code and med_ingr_rxnorm_code.
# I queried RxMix with strings for their ingredient codes.
# Two things:
# for rows with rxnorm, I queried with rxnorm for ingredients, and I will use mine unless it does not exist. 
# for rows without rxnorm, I queried with strings.
# mechanism of action might be useful, but I have no such confidence, and unique ingredients are fewer than I assumed

# I chose to get an XML table because it preserves the structure.
# this means you need to know XPath, otherwise parsing is a pain.
require(xml2)
require(XML)

# readable print from XML
 
readableTree<-xmlParse('/infodev1/home/m193194/git/ehr/death/data/missing_string_approx_ingr_xml/05270cb1aaede71a0bc348d9d0c8ef5f.xml')
xmldoc<-read_xml('/infodev1/home/m193194/git/ehr/death/data/missing_string_approx_ingr_xml/05270cb1aaede71a0bc348d9d0c8ef5f.xml')
inputs<-xml_find_all(xmldoc,"function/input")
calls<-xml_find_all(xmldoc,"./function")
# use XPath
# this is very slow without parallel
# find first returns error
# there is no way we can do this without core dumps. See below. I tried mclapply foreach and others. No way.
medrxcui<-lapply(inputs,function(x) xml_find_all(x,"./following-sibling::outputs/output/RXCUI"))
ingr_cui<-lapply(calls,function(x) xml_find_all(x,".//function[@level='1']//output/RXCUI"))
chosenmedrx<-lapply(calls,function(x) xml_find_all(x,".//function[@level='1']//input"))

# commented out
if (False){
    # glibc throws core dumps
    # running parallel on this xml2 xmlnodeset object seems to be very problematic
    # the objects use "externalpointers"?
    library(doParallel)
    library(foreach)
    cl<-makeCluster(16)
    registerDoParallel(cl)

    by_group<-split(inputs,rep(1:16,length.out=length(inputs)))
    medrxcui<-foreach(tt=by_group, .packages=c('xml2') )%dopar% {
        lapply(tt, function(x) xml_find_all(x,"./following-sibling::outputs/output/RXCUI"))}
    stopCluster(cl)

    cl<-makeCluster(16)
    registerDoParallel(cl)
    inputs<- inputs %>% mutate(group=rep(1:16,length.out=nrows(inputs))) %>% setDT()
    by_group<-split(inputs,by="group",keep.by=F)
    medrxcui<-foreach(tt=by_group,.combine=rbind, .packages=c('dplyr','data.table') )%dopar% {
        tt %>%
        lapply(function(x) xml_find_first(x,"./following_sibling:outputs/output/RXCUI")) %>%
        setDT()}
    stopCluster(cl)
}

# dump results. cannot be dumped as robject
#fwrite(xml_text(inputs)%>%as.list(),'/infodev1/rep/projects/jason/parsed_inputs.csv')
hello<-lapply(medrxcui, function(x) { xml_text(x)[1] })
#fwrite(hello %>% as.list,'/infodev1/rep/projects/jason/parsed_medrxcui.csv')
hello<-list()
hello<-lapply(1:length(inputs), function(x) { hello[[x]]<-xml_text(ingr_cui[[x]])  })
#saveRDS(hello,file='/infodev1/rep/projects/jason/parsed_ingr_rxcui.rds')
chosen<-c()
chosen<-lapply(1:length(inputs), function (x) {chosen<-c(chosen,xml_text(chosenmedrx[[x]])[1])})
#fwrite(chosen, '/infodev1/rep/projects/jason/parsed_first_queryed_medrxcui.csv')

# we encounter a problem.
# the medicine to ingredient is not a one to one mapping. This means we will not be able to store it efficiently in data frame without chopping off, we even have difficulty storing it in a csv table.
# to do so, I will write the file manually. We will populate med_rxnorm in the prescription file. We will store med_rxnorm to ingr_rxnorm mapping in another file.
# we will use a bar and comma separation scheme. e.g. 1922med91| 19ingr184, 110ingr1924, 182ingr165

# we first run further cleaning, since I found in my query that there are some dirty rows still
mypres<-mypres %>% select(rep_person_id,MED_DATE,med_name,med_generic,med_rxnorm_code,med_ingr_rxnorm_code)
fwrite(mypres,"/infodev1/rep/projects/jason/mypres_temp.csv")
mypres<- mypres %>% mutate(med_rxnorm_code=if_else(med_rxnorm_code=="","0",med_rxnorm_code))%>%mutate(med_rxnorm_code=as.integer(med_rxnorm_code)) %>% filter(!is.na(med_rxnorm_code)) %>% setDT()
mypres<- mypres %>% mutate(med_ingr_rxnorm_code=if_else(med_ingr_rxnorm_code=="", "0", med_ingr_rxnorm_code)) %>% mutate(med_ingr_rxnorm_code=as.integer(med_ingr_rxnorm_code)) %>% filter(!is.na(med_ingr_rxnorm_code)) %>% setDT()

# we now populate the missing rxnorms in mypres
chosen<-unlist(chosen)
names<-xml_text(inputs)
nametorxnorm<-data.table(name=names,rxnorm=chosen)
try <- mypres%>% left_join(nametorxnorm,by=c("med_name"="name")) %>% mutate(rxnorm=as.integer(rxnorm)) %>%setDT()
try <- try %>% mutate(med_rxnorm_code=if_else(med_rxnorm_code==0,rxnorm,med_rxnorm_code)) %>% setDT()
# these are really empty rows
try <- try[!is.na(med_rxnorm_code)] 
# you will see, there are still dirty data in columns we don't care, but I can guarantee by type that the ones we have should be cleaned.
try <- try %>% arrange(rep_person_id,MED_DATE) %>% setDT()
try[med_rxnorm_code==0]$med_rxnorm_code<-NA
try[med_ingr_rxnorm_code==0]$med_ingr_rxnorm_code<-NA
fwrite(try,"/infodev1/rep/projects/jason/verbose_mypres.csv")
try_min <- try %>% select(-med_name,-med_generic,-rxnorm, -med_ingr_rxnorm_code) %>% setDT()
try_min[med_rxnorm_code==0]$med_rxnorm_code<-NA
fwrite(try_min,'/infodev1/rep/projects/jason/min_mypres.csv')

# we need to create the bar comma file for mapping from med_rxnorm_code to ingr_rxnorm_code
# two lists
# list of (existing) name, rxnorm, ingrdients mapping
readableTree<-xmlParse('/infodev1/home/m193194/git/ehr/death/data/aug_rxnorm_to_ingr/b2e6a98552d25b4b7772b6cc557dc65e.xml')  
xmldoc<-read_xml('/infodev1/home/m193194/git/ehr/death/data/aug_rxnorm_to_ingr/b2e6a98552d25b4b7772b6cc557dc65e.xml')  
calls<-xml_find_all(xmldoc,"./function") 
lhs<-lapply(calls,function(x) {
            y<-xml_find_all(x,"./input")
            xml_text(y)
}
)
rhs<-lapply(calls,function(x){
            y<-xml_find_all(x,".//output/RXCUI")
            xml_text(y)
})

# list of (nonexisting) rxnorm, ingrdients mapping
# left: chosen
lhs2<-chosen
# right:
rhs2<-lapply(ingr_cui,xml_text)

# longlist
lc<-c(lhs,lhs2)
rc<-c(rhs,rhs2)
saveRDS(lc,file='/infodev1/rep/projects/jason/lc.rds')
saveRDS(rc,file='/infodev1/rep/projects/jason/rc.rds')

try<-data.table(lc=lc,rc=rc)
try <- try %>% mutate(lc=unlist(lc)) %>% setDT()
try <- try %>% distinct(lc,.keep_all=T) %>% setDT()
fwrite(try,'/infodev1/rep/projects/jason/rxnorm_ingr_dt.csv')

# We need to manually examine the results to see if the conversion is sound.
mypres <- mypres %>% left_join(lookup,by=c(med_rxnorm_code='lc')) %>% setDT()
# We've come a long way here, and I cannot rule out any mistake.
# From the conversion table, we have many ingrdients fields missing.
# I want to ensure that if med_ingr_rxnorm_code exists in the original database and does not get result in the query, then the final table will have its original med_ingr_rxnorm_code
original_lookup<-mypres%>% select(med_rxnorm_code,med_ingr_rxnorm_code) %>% setDT()
original_lookup<-original_lookup[!is.na(med_ingr_rxnorm_code)]
original_lookup<- original_lookup %>% distinct(med_rxnorm_code,.keep_all=T) %>% setDT()

# lookup is read from the file
lookup<-fread('/infodev1/rep/projects/jason/rxnorm_ingr_dt.csv')
lookup<- lookup%>%mutate(rc=unlist(rc)) %>% setDT()
lookup<- lookup[rc!=""]
notin<-! original_lookup$med_rxnorm_code %in% lookup$lc
notin<-original_lookup[notin]
colnames(notin) <- colnames(lookup)
lookup<-rbind(lookup,notin)
lookup <- lookup %>% arrange(lc) %>% setDT()
fwrite(lookup,'/infodev1/rep/projects/jason/new_rxnorm_ingr_dt.csv')

# our lookup is very good, and we examine again and find that some rows are obsolete.
# You can test by running mypres[med_generic=="ASPIRIN"], some rows use obsolete med_rxnorm_code and therefore has no result for med_ingr_rxnorm_code or anything
# we will need to deal with such rows.

# Condition:
# is.na(rc)
# med_name or med_generic exists in our database, local or lookup
# run local search so that the med_rxnorm_code is corrected

failsafe<-mypres[,c("med_name","med_rxnorm_code")]
failsafe<- failsafe %>% group_by(med_name,med_rxnorm_code) %>% mutate(n=n()) %>% setDT()
failsafe <- failsafe %>% group_by(med_name) %>% mutate(maxn=max(n)) %>% setDT()
failsafe <- failsafe %>% filter(n==maxn) %>% select (-n) %>% setDT()
failsafe<- failsafe %>% distinct(med_name,.keep_all=T) %>% setDT()
failsafe<- failsafe %>% select(-maxn) %>% filter(!is.na(med_rxnorm_code)) %>% setDT()

# now we REPLACE med_rxnorm_code for all those who don't have rc
mypres[is.na(rc)]
failsafe<- failsafe %>% mutate(new_med_rxnorm=med_rxnorm_code) %>% select(-med_rxnorm_code) %>% setDT()
mypres <- mypres %>% left_join(failsafe) %>% setDT()
mypres <- mypres %>% mutate(med_rxnorm_code=if_else(is.na(rc),new_med_rxnorm,med_rxnorm_code)) %>% setDT()
mypres <- mypres %>% select(-new_med_rxnorm,-rc) %>% setDT()

# forget the bar and comma. it turns out that we can store bar string in csv files. that's how it should be done, natively without another dict file
fwrite(mypres,'/infodev1/rep/projects/jason/before_drop_mypres.csv')
mypres <- mypres %>% select(-med_ingr_rxnorm_code,-queried_med_rxnorm) %>% mutate(med_ingr_rxnorm_code=rc) %>% select(-rc) %>% setDT()
fwrite(mypres,'/infodev1/rep/projects/jason/new_verbose_rxnorm_ingr_dt.csv')
mypres <- mypres %>% select(-med_name,-med_generic,-med_rxnorm_code) %>% setDT()
fwrite(mypres,'/infodev1/rep/projects/jason/mypres.csv')

# we have finished

###### SERVICES
# my intuition tells me that sevices will not beo too vital
# I will filter our the tail of the dataset to control input complexity.
serv<-fread('/infodev1/rep/data/services.dat')
#services_table <- serv %>% select(srv_px_code) %>% group_by(srv_px_code) %>% mutate(count=n()) %>% distinct(srv_px_code, .keep_all=TRUE) %>%  arrange(count) %>%  setDT()
#collapse<-services_table[count<1000]$srv_px_code
myserv<-serv %>% select (rep_person_id, SRV_DATE, srv_px_count, srv_px_code, SRV_LOCATION,  SRV_ADMIT_DATE, srv_admit_type, srv_admit_src, SRV_DISCH_DATE, srv_disch_stat)
#myserv[srv_px_code %in% collapse]$srv_px_code<-"other"
myserv<-myserv %>% mutate(SRV_DATE=mdy(SRV_DATE),SRV_ADMIT_DATE=mdy(SRV_ADMIT_DATE), SRV_DISCH_DATE=mdy(SRV_DISCH_DATE)) %>% setDT()
#services_table<- myserv %>% select(srv_admit_type) %>% group_by(srv_admit_type) %>% mutate(n=n()) %>% distinct(srv_admit_type, .keep_all=T) %>% setDT()
#collapse<-services_table[n<1000]$srv_admit_type
#myserv[srv_admit_type %in% collapse]$srv_admit_type<-"other"

coll<-function(dt,colname,thres,replace_value){
    st<- dt %>% select_(colname) %>% group_by_(colname) %>% mutate(n=n()) %>% distinct_(colname,.keep_all=T) %>% setDT()
    collapse<-st[n<thres][[colname]]
    dt[get(colname) %in% collapse][[colname]] <-replace_value
    dt
}
myserv<-coll(myserv,"srv_px_code",1000,"other")
myserv<-coll(myserv,"srv_admit_type",1000,"other")
myserv<-coll(myserv,"srv_admit_src",1000,"other")
myserv<-coll(myserv,"srv_disch_stat",1000,"other")
myserv<-coll(myserv,"srv_px_count",500,1L)
myserv[srv_px_count==-2|srv_px_count==-1]$srv_px_count<-1
myserv<-coll(myserv,"SRV_LOCATION",1000,"other")

# By running commands like myserv[1:10000][SRV_DATE==SRV_ADMIT_DATE] we see that the SRV_DATE is usually highly associated with the admission date and dispatch date. but three dates would triple the inputs
# an admission can have multiple serrvices, so that complicate our inputs
# one date per row.
myserv <- myserv %>% select(-SRV_ADMIT_DATE,-SRV_DISCH_DATE) %>% setDT()
# bar separated is not possible here
myserv[is.na(srv_px_count)]$srv_px_count<-1


fwrite(myserv,"/infodev1/rep/projects/jason/myserv.csv")
fwrite(as.list(unique(myserv$srv_px_code)),"/infodev1/rep/projects/jason/myserv_all_px_codes")

######## SURGERIES
surg<-fread("/infodev1/rep/data/surgeries.dat")
mysurg<-surg%>%select(rep_person_id,px_date,px_codetype,px_code) %>%setDT()
# I am going to conver all I9 codes to ICD10 codes here. For table: https://www.cms.gov/medicare/coding/ICD10/2014-ICD-10-PCS.html
pcs<-fread("/home/m193194/git/ehr/death/data/gem_pcsi9.txt")
colnames(pcs)<-c("i10","i9","flag")
pcs<- pcs %>% distinct(i10,.keep_all=T)
# fixed a bug here for multiple matches
mysurg <- mysurg %>% left_join(pcs,by=c("px_code"="i10")) %>% setDT()
# no dot
mysurg <- mysurg %>% separate(px_code,c("first","second"),remove=FALSE) %>%  setDT()
mysurg <- mysurg%>% unite("nodot",c("first","second"),sep="") %>% setDT()
mysurg <- mysurg %>% mutate(nodot=as.integer(nodot)) %>% setDT()
mysurg <- mysurg %>% mutate(px_code=if_else(px_codetype=="I9",nodot,i9)) %>% setDT()
mysurg <- mysurg %>% select(-nodot, -i9, -flag, -px_codetype) %>% setDT()
mysurg <- mysurg %>% mutate(px_date=mdy(px_date)) %>% setDT()
mysurg <- mysurg[!is.na(px_code)] %>% mutate(rep_person_id=as.integer(rep_person_id)) %>% setDT()

# ICD9 is much better.
# we eliminate tail.
# The algorithm is simple. We see the count for each 4 digit code, and if the count is fewer than 2000, then we aggregate them to a 3 digit code
# I can do it because ICD code is structured
# 2000 and 4 are arbitrary decisions
mysurg <- mysurg %>% group_by(px_code) %>% mutate(n=n()) %>% setDT()
mysurg <- mysurg %>% mutate(other=n<2000) %>% setDT()
mysurg <- mysurg %>% mutate(collapsed_px_code=if_else(other==T,as.integer(px_code%/%10),px_code)) %>% setDT()
# I think it's worth it. The dimension has been collapsed to under 1000, compared to 20000.
mysurg <- mysurg%>% select(-n,-other)
mysurg <- mysurg %>% arrange(rep_person_id,px_date) %>% setDT()
fwrite(mysurg,"/infodev1/rep/projects/jason/mysurg.csv")

######## TOBACCO
# This file cannot be processed at the moment.
# we need careful NLP feature engineering and extraction
# or we need actual tobacco labels. Well, we might be able to extract it from our EHR, but that's for another day.

######## VITALS
# for vitrals, we remove positions, becuase those things are crazy.
vitals<-fread('/infodev1/rep/data/vitals.dat')
vitals<-vitals%>% select(-VITAL_VALUE_TXT,-vital_seq,-vital_src,-vital_src_code,-VITAL_SRC_DESC) %>% setDT()
vitals<-vitals[vital_name!="BP POSITION"]
# this dataset is rather clean. We don't need to do much.
vitals<-vitals[!is.na(vitals$vital_value_num)]
# we will do a pivot table and adhere to our person_date uniqueness, becuase it's possible here.
# but before that, let's convert the units to metric, needed before reshape.
# for BP, it's mmHg
# for height, it's cm
# for weight, it's kg
# use this line to confirm that a unit is unique to a vital_name
# res<-lapply(unique(vitals$vital_name),function(x) table(vitals[vital_name==x,VITAL_UNIT]))
# no better way to do it but manual
vitals<-vitals %>% mutate(vital_value_num=if_else(VITAL_UNIT=="lb"|VITAL_UNIT=="LBS",vital_value_num*0.453592,vital_value_num)) %>% setDT()
vitals<-vitals %>% mutate(vital_value_num=if_else(VITAL_UNIT=="inch(es)",vital_value_num*2.54,vital_value_num)) %>% setDT()
# see change: vitals[VITAL_UNIT=="LBS"|VITAL_UNIT=="lb"|VITAL_UNIT=="inch(es)"]
vitals<-vitals %>% select(-VITAL_UNIT) %>% setDT()

### now we pivot
#require(reshape2)
## let's parse the date first, in case equivalence is required
vitals<-vitals%>% mutate(VITAL_DATE=substr(VITAL_DATE,1,9))%>%setDT()
vitals<-vitals%>% mutate(VITAL_DATE=dmy(VITAL_DATE)) %>% setDT()
## hello<-cast(vitals,rep_person_id+VITAL_DATE ~ vital_name)
## hello<-hello %>% setDT()
## as usual, pivot table is very slow, so we are going to run this in parallel
#library(parallel)
#cl<-32
#group<-rep(1:cl,length.out=nrow(vitals))
#vitals<-vitals%>%mutate(group=rep(1:cl,length.out=nrow(vitals))) %>% setDT()
#require(multidplyr)
#cluster<-create_cluster(cores=cl)
#by_group <- vitals %>% partition(group,cluster=cluster)
#by_group %>%
#    # Assign libraries
#    cluster_library("reshape2") %>%
#    cluster_library("data.table") %>%
#    cluster_library("dplyr")
#    # Assign values (use this to load functions or data to each core)
#start <- proc.time() # Start clock
#vitals_pivot_parallel<-by_group %>% dcast(rep_person_id+VITAL_DATE ~ vital_name,fun.aggregate=mean) %>% collect() %>% setDT()
#time_elapsed<-proc.time()-start
## reshape/reshap2 are not compatible with multidplyr

#library(parallel)
#no_cores<-16
#cl<-makeCluster(no_cores)
vitals<-vitals%>%mutate(group=rep(1:no_cores,length.out=nrow(vitals))) %>% setDT()
by_group<-split(vitals,by="group",keep.by=F)
#clusterExport
#res<-parLapply(cl,by_group,function(table) {
#               dcast(rep_person_id+VITAL_DATE ~ vital_value_num, mean) %>%
#               setDT()})

# there are a few measurements in one day. I don't really get why.
# for our purpose, we will use a daily precision
library(doParallel)
library(foreach)
cl<-makeCluster(16)
registerDoParallel(cl)
res<-foreach(tt=by_group,.combine=rbind, .packages=c('dplyr','data.table') )%dopar% {
    tt %>%
    dcast(rep_person_id+VITAL_DATE ~ vital_name, value.var="vital_value_num",fun.aggregate=mean) %>%
    setDT()}
stopCluster(cl)
res <- res%>%arrange(rep_person_id,VITAL_DATE) %>% setDT()
fwrite(res,'/infodev1/rep/projects/jason/myvitals.csv')

# DEMOGRAPHICS
# requires reading all previous files

demo<-fread('/infodev1/rep/data/demograph')
# age is thrown away. Difficulty to process. Obfuscate valuable signal.
demo <- demo %>% select(rep_person_id,sex,race,ethnicity,educ_level,birth_date)
demo <- demo %>% mutate(male=sex=="M") %>% setDT()
demo <- demo %>% select (-sex)
demo <- demo %>% arrange(rep_person_id) %>% setDT()
# not sure what it is, only 13870 cases, binary
demo <- demo %>%select(-ethnicity) %>% setDT()
demo <- demo %>% mutate(birth_date=mdy(birth_date)) %>% setDT()
demo <- demo[!is.na(birth_date)]

# remove people with no inputs
death<-fread('/infodev1/rep/projects/jason/deathtargets.csv')
demo<-fread('/infodev1/rep/projects/jason/demo.csv')
dia<-fread('/infodev1/rep/projects/jason/mydia.csv')
hos<-fread('/infodev1/rep/projects/jason/myhosp.csv')
lab<-fread('/infodev1/rep/projects/jason/mylabs.csv')
pres<-fread('/infodev1/rep/projects/jason/mypres.csv')
serv<-fread("/infodev1/rep/projects/jason/myserv.csv")
surg<-fread("/infodev1/rep/projects/jason/mysurg.csv")
vitals<-fread("/infodev1/rep/projects/jason/myvitals.csv")

ldf=list(dia,hos,lab,pres,serv,surg,vitals)
notin<-copy(demo)
for (df in ldf){
    notin <- notin[!rep_person_id %in% df$rep_person_id]
}
demo<-demo[!rep_person_id %in% notin$rep_person_id]

fwrite(demo,'/infodev1/rep/projects/jason/demo.csv')
