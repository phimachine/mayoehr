# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 2/3/2019


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
hic<-fread('data/hicda.csv')



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
####### This is where the conversion goes backwards. 9 to 10 instead

gem<-fread('data/2018_I9gem.txt')
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
fwrite(main,"/infodev1/rep/projects/jason/new/deathtargets.csv")
