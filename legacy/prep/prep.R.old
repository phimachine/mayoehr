# okay, let's do it again.
# this time, we will adhere to a month of precision, because the date is easy to parse

require(lubridate)
require(data.table)
require(dplyr)
require(doParallel)
require(tidyr)
require(fuzzyjoin)


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


# no dot
main <- main%>% separate(code,c("first","second"),remove=FALSE) %>%  setDT()
main <- main %>% replace_na(list(first="",second="")) %>% setDT()
main <- main%>% unite("nodot",c("first","second"),sep="") %>% setDT()
main <- main%>% mutate(nodot=if_else(code_type=="ICD10",nodot,"")) %>% setDT()





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
mydia<-mydia%>%mutate(dx_date=mdy(dx_date))%>%setDT()
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
mysurg<-mysurg%>% select(-n,-other)
fwrite(mysurg,"/infodev1/rep/projects/jason/mysurg.csv")

######## 
