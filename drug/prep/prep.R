# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 2/3/2019


# this is a rework that enforces icd9->icd 10 conversion instead. Does it improve my dataset quality?
# I only modify the datasets that contain icd codes: death, diagnosis, hospitalization, surgeries.


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
main <- main %>% mutate(code_type=if_else(code_type=="HIC","ICD9",code_type)) %>% setDT()
####### This is where the conversion goes backwards. 9 to 10 instead

gem<-fread('data/2018_I9gem.txt')
colnames(gem)<-c("i9","i10","flags")
# this gem file has missing rows, and it's going to be problematic. We need to manually generate more conversions by guessing it.
# we probably don't need to do it in python. We have fuzzyjoin library.
require(fuzzyjoin)
# no dot, because our conversion table
main <- main%>% separate(code,c("first","second"),remove=FALSE) %>%  setDT()
main <- main %>% replace_na(list(first="",second="")) %>% setDT()
main <- main%>% unite("nodot",c("first","second"),sep="") %>% setDT()

main <- main%>% mutate(nodot=if_else(code_type=="ICD9",nodot,"")) %>% setDT()



# potentially possible for parallel computation, this takes forever
start <- proc.time() # Start clock
#1114
icd9_lines<- main%>% filter(code_type=="ICD9") %>% setDT()

regex_result <- gem %>% regex_right_join(icd9_lines,by=c(i9="nodot")) %>% setDT()
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
#976
regex_result <- regex_result %>% mutate(code=i10) %>% setDT() %>% select(-i9, -i10, -flags)
main <- main %>% filter(code_type=="ICD10") %>% setDT()
main <- rbind(main, regex_result)
#90153


# HIC still has dots. remove
main <- main%>% separate(code,c("first","second"),remove=FALSE) %>%  setDT()
main <- main %>% replace_na(list(first="",second="")) %>% setDT()
main <- main%>% unite("nodot",c("first","second"),sep="") %>% setDT()
main <- main %>% mutate(code=if_else(code_type=="ICD10",nodot,code)) %>% select(rep_person_id,death_date, underlying,code_type,code) %>% setDT()
#90153

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
hic<-fread('data/hicda.csv')
colnames(hic)<-c("hicda","hicda_desc","icd9","icd9_desc","grpnbr")
# transform the hic conversion table for our application..
hello <- hic %>% distinct(hicda, .keep_all=T) %>% setDT()
hello <- hello %>% select(hicda,icd9) %>% setDT()
hello <- hello %>% mutate(hicda=as.character(hicda)) %>% setDT()
hello<- hello %>% mutate(hicda=str_pad(hicda,8,pad="0")) %>% setDT()
mydia<- mydia %>% left_join(hello,by=c("DX_CODE"="hicda")) %>% setDT()
mydia<- mydia %>% mutate(DX_CODE=if_else(dx_codetype=="HIC", icd9, DX_CODE)) %>% select(-icd9) %>% setDT()
mydia<- mydia %>% mutate(dx_codetype=if_else(dx_codetype=="HIC","I9",dx_codetype)) %>% setDT()
# transform I9 to I10 instead
gem<-fread('data/2018_I9gem.txt')
colnames(gem)<-c("i9","i10","flag")
gem<- gem %>% distinct(i9,.keep_all=T) %>% select(-flag) %>% setDT()

mydia<- mydia %>% separate(DX_CODE,c("first","second"), remove=F)
mydia <- mydia %>% replace_na(list(first="",second="")) %>% setDT()
mydia <- mydia %>% unite("nodot", c("first","second"),sep="") %>% setDT()

# do not modify
# ret<-fread("/infodev1/rep/projects/jason/mydiag_rett.csv")
# for the regex join, we will need to run it efficiently.
all_i9_codes <- mydia %>% filter(dx_codetype=="I9") %>% distinct(nodot) %>% setDT()
# all_i9_codes <- all_i9_codes %>% mutate(nodot=paste("^",nodot,sep="")) %>% setDT()


library(foreach)
library(parallel)
cl<-parallel::makeForkCluster(32)
doParallel::registerDoParallel(cl)
ret2<-foreach(i=1:nrow(all_i9_codes), .combine='rbind') %dopar%{
    gem%>% right_join(all_i9_codes[i],by=c(i9="nodot"))}
stopCluster(cl)

ret2 <- ret2 %>% setDT()
second_run <- ret2[is.na(i10)]
# after a few samples, I think the second run consists of two types: those that missed a 0 at the end, and those that
# are actually icd10 codes.
# first we append zero and search again
second_run_type_A<-second_run %>% mutate(i9=paste(i9,"0",sep="")) %>% select(-i10) %>% setDT()

cl<-parallel::makeForkCluster(8)
doParallel::registerDoParallel(cl)
second_run_type_A_ret<-foreach(i=1:nrow(second_run_type_A), .combine='rbind') %dopar%{
    gem%>% right_join(second_run_type_A[i],by=c(i9="i9"))}
stopCluster(cl)

second_run_type_A_ret <- second_run_type_A_ret %>% setDT()
second_run_type_A_ret <- second_run_type_A_ret %>% mutate(i9=second_run$i9) %>% setDT()
# 1230 rows found their matches
second_run_type_A_ret <-second_run_type_A_ret[!is.na(i10)]



second_run_type_B<- second_run %>% filter(!i9 %in% second_run_type_A_ret$i9) %>% setDT()
# many of them are i9 codes that are not specific enough to be converted. These codes are not touched.
# I am looking for exact matches of icd 10 codes, or else they will be considered no match.
second_run_type_B <- second_run_type_B %>% filter(i9 %in% gem$i10) %>% setDT()
second_run_type_B_ret <- second_run_type_B %>% mutate(i10=i9) %>% setDT()

second_run_type_C <- second_run %>% filter(!i9 %in% second_run_type_A_ret) %>% filter(! i9 %in% second_run_type_B_ret) %>% setDT()
second_run_type_C <-second_run_type_C %>% mutate(i9=paste(i9,"1",sep="")) %>% select(-i10) %>% setDT()

cl<-parallel::makeForkCluster(8)
doParallel::registerDoParallel(cl)
second_run_type_C_ret<-foreach(i=1:nrow(second_run_type_C), .combine='rbind') %dopar%{
    gem%>% right_join(second_run_type_C[i],by=c(i9="i9"))}
stopCluster(cl)

second_run_type_C_ret <- second_run_type_C_ret %>% setDT()
second_run_type_C_ret <- second_run_type_C_ret %>% mutate(i9=second_run$i9) %>% setDT()
second_run_type_C <- second_run %>% filter(!i9 %in% second_run_type_A_ret) %>% filter(! i9 %in% second_run_type_B_ret) %>% setDT()
second_run_type_C_ret <- second_run_type_C_ret %>% mutate(i9=second_run_type_C$i9)


# 1230 rows found their matches
second_run_type_C_ret<-second_run_type_C_ret[!is.na(i10)]

ret2 <- ret2 %>% filter(!is.na(i10)) %>% setDT()
ret2 <- rbind(ret2, second_run_type_A_ret, second_run_type_B_ret, second_run_type_C_ret)
ret2 <- ret2 %>% filter(!is.na(i10)) %>% filter(i10!="NoDx")%>% setDT()


fwrite(ret2,"/infodev1/rep/projects/jason/new/mydia_ret.csv")

# Throws error if 48 clusters
# Error in unserialize(socklist[[n]]) : error reading from connection
#fwrite(ret,"/infodev1/rep/projects/jason/mydia_reg")

# this is because we do matches on i9 codes too, and they have mutliple matches. This piece of code is very bad. We could've run the regex match then leftjoin
mydia_i9<- mydia %>% filter(dx_codetype=="I9") %>% setDT()
mydia_i10 <- mydia %>% filter(dx_codetype=="I10") %>% setDT()

# very fast join on 10% of the dataset with no fuzzy match.
mydia_i9 <- mydia_i9 %>%  left_join(ret2,by=c("nodot"="i9")) %>% setDT()
mydia_i9 <- mydia_i9 %>% mutate(nodot=i10) %>% select(-i10) %>% setDT()

# lost 3% of the data in this process, given that 96% of the dataset is ICD9
mydia <- rbind(mydia_i9, mydia_i10) %>% setDT()
alli10<-fread('data/all_icd10.txt')
# lost another 5 here. Rid of the NAs
mydia <- mydia %>% filter(nodot %in% alli10$code) %>% setDT()

mydia <- mydia %>% select(-dx_codetype,-DX_CODE,-dx_code_seq) %>% setDT()
# 60  is distinct values
mydia <- mydia %>% distinct(rep_person_id,dx_date,nodot) %>% setDT()
# 58353148
mydia <- mydia %>% arrange(rep_person_id, dx_date) %>% setDT()

fwrite(mydia,"/infodev1/rep/projects/jason/new/mydia_before_bar.csv")

#fwrite(as.list(unique(mydia$DX_CODE)),"/infodev1/rep/projects/jason/mydia_all_dx_codes")
# no bar sepr. the loading time at inputgen needs to be improved. String operations are expensive.

# # I like bar separated notations
# mydia <- mydia %>% group_by(rep_person_id,dx_date) %>% mutate(dx_codes=paste0(nodot,collapse="|")) %>% setDT()
# mydia <- mydia %>% distinct(rep_person_id,dx_date,.keep_all=T) %>% setDT()
# mydia <- mydia%>% mutate(dx_date=mdy(dx_date)) %>% setDT()
#fwrite(mydia,"/infodev1/rep/projects/jason/mydia_no_distinct.csv")
# mydia <- mydia %>% distinct(rep_person_id,dx_date,dx_codes) %>% setDT()
# mydia <- mydia %>% arrange(rep_person_id, dx_date) %>% setDT()
#fwrite(mydia,"/infodev1/rep/projects/jason/mydia.csv")


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

gem<-fread('data/2018_I9gem.txt')
colnames(gem)<-c("i9","i10","flag")
gem<- gem %>% distinct(i9,.keep_all=T) %>% select(-flag) %>% setDT()

vi9<-c()
myhosp_i9<- myhosp %>% filter(hosp_dx_code_type=="I9") %>% setDT()


# collect all unique i10, so that regex join does not go through all rows.
##### THIS FUNCTION COLLECTS NOT ONLY I10 BUT ALSO I9, so we cater to it later
collect_i9<- function(myhosp,key,vi10){
    myhosp <- myhosp %>% separate(key,c("first","second"), remove=F)
    myhosp <- myhosp %>% replace_na(list(first="",second="")) %>% setDT()
    myhosp <- myhosp %>% unite("nodot", c("first","second"),sep="") %>% setDT()
    vi9<-c(vi9,myhosp$nodot)
    vi9
}
# select all expanded columns
keys<- myhosp[1:2] %>% select(starts_with("hosp_secondary_dx")) %>% colnames()
keys<-c("hosp_primary_dx",keys)
# collect all i10 codes
for (key in keys) {
    print(key)
    vi9<-collect_i9(myhosp_i9,key,vi9)
}


# convert all i10 codes. Two passes are more efficient
uvi9<-unique(vi9)
uvidt<-data.table(i9=uvi9)

# this line does not run, what does it mean?
uvidt<-uvidt[i9!=""]

# should take around 1 hour with this configuration
cl<-parallel::makeForkCluster(8)
doParallel::registerDoParallel(cl)
ret<-foreach(i=1:nrow(uvidt),.combine='rbind') %dopar%{
    gem %>% right_join(uvidt[i],by=c("i9"="i9"))
}
stopCluster(cl)
# lost 3%
ret <- ret %>% setDT()

second_run <- ret[is.na(i10)]
# after a few samples, I think the second run consists of two types: those that missed a 0 at the end, and those that
# are actually icd10 codes.
# first we append zero and search again
second_run_type_A<-second_run %>% mutate(i9=paste(i9,"0",sep="")) %>% select(-i10) %>% setDT()

cl<-parallel::makeForkCluster(8)
doParallel::registerDoParallel(cl)
second_run_type_A_ret<-foreach(i=1:nrow(second_run_type_A), .combine='rbind') %dopar%{
    gem%>% right_join(second_run_type_A[i],by=c(i9="i9"))}
stopCluster(cl)

second_run_type_A_ret <- second_run_type_A_ret %>% setDT()
second_run_type_A_ret <- second_run_type_A_ret %>% mutate(i9=second_run$i9) %>% setDT()
# 1230 rows found their matches
second_run_type_A_ret<-second_run_type_A_ret[!is.na(i10)]


second_run_type_B<- second_run %>% filter(!i9 %in% second_run_type_A_ret$i9) %>% setDT()
# many of them are i9 codes that are not specific enough to be converted. These codes are not touched.
# I am looking for exact matches of icd 10 codes, or else they will be considered no match.
second_run_type_B <- second_run_type_B %>% filter(i9 %in% gem$i10) %>% setDT()
second_run_type_B_ret <- second_run_type_B %>% mutate(i10=i9) %>% setDT()

second_run_type_C <- second_run %>% filter(!i9 %in% second_run_type_A_ret) %>% filter(! i9 %in% second_run_type_B_ret) %>% setDT()
second_run_type_C <-second_run_type_C %>% mutate(i9=paste(i9,"1",sep="")) %>% select(-i10) %>% setDT()

cl<-parallel::makeForkCluster(8)
doParallel::registerDoParallel(cl)
second_run_type_C_ret<-foreach(i=1:nrow(second_run_type_C), .combine='rbind') %dopar%{
    gem%>% right_join(second_run_type_C[i],by=c(i9="i9"))}
stopCluster(cl)

second_run_type_C_ret <- second_run_type_C_ret %>% setDT()
second_run_type_C_ret <- second_run_type_C_ret %>% mutate(i9=second_run$i9) %>% setDT()
second_run_type_C <- second_run %>% filter(!i9 %in% second_run_type_A_ret) %>% filter(! i9 %in% second_run_type_B_ret) %>% setDT()
second_run_type_C_ret <- second_run_type_C_ret %>% mutate(i9=second_run_type_C$i9)


# 1230 rows found their matches
second_run_type_C_ret<-second_run_type_C_ret[!is.na(i10)]

ret2 <- ret %>% filter(!is.na(i10)) %>% setDT()
ret2 <- rbind(ret2, second_run_type_A_ret, second_run_type_B_ret, second_run_type_C_ret)
ret2 <- ret2 %>% filter(!is.na(i10)) %>% filter(i10!="NoDx")%>% setDT()


#fwrite(ret,"/infodev1/rep/projects/jason/myhosp_reg")
# replace all i9!!! with the corresponding regex result.
require(reshape2)
myhosp2<-melt(myhosp, id.vars=c("rep_person_id","hosp_admit_dt","hosp_disch_dt","hosp_inout_code","hosp_adm_source","hosp_disch_disp",
"hosp_dx_code_type"))
myhosp2<-myhosp2%>% select(-variable) %>% filter(value!="") %>% setDT()

# 3958662
#
# # normal left join
myhosp2<-myhosp2 %>% separate(value,c("first","second"), remove=F) %>% replace_na(list(first="",second="")) %>% setDT()
myhosp2 <- myhosp2%>% unite("nodot", c("first","second"),sep="") %>% left_join(ret, by=c("nodot"="i9")) %>% setDT()
#
# batch_to_i9 <- function(myhosp,key,regret){
#     myhosp <- myhosp %>% separate(key,c("first","second"), remove=F)
#     myhosp <- myhosp %>% replace_na(list(first="",second="")) %>% setDT()
#     myhosp <- myhosp %>% unite("nodot", c("first","second"),sep="") %>% setDT()
#     myhosp <- myhosp %>% left_join(regret, by=c("nodot"="uvi10i9")) %>% setDT()
#     # i9 codes have nonsensical matches, but they will not be used here
#     myhosp <- myhosp %>% mutate(!!key:=if_else(hosp_dx_code_type=="I10", i9, nodot)) %>% setDT()
#     myhosp <- myhosp %>% select(-i9) %>% setDT()
#     myhosp
# }
#
#
# # combine back
# for (key in keys){
#     print(key)
#     myhosp<-batch_to_i9(myhosp,key,ret)
# }

myhosp2<- myhosp2 %>% mutate(nodot=if_else(hosp_dx_code_type=="I9",i10,nodot)) %>% setDT()
myhosp2<- myhosp2 %>% select(-value, -i10, -hosp_dx_code_type) %>% setDT()
# MANUALLY EXAMINE
hosp<- hosp %>% arrange(rep_person_id, hosp_admit_dt) %>% setDT()
myhosp2<- myhosp2 %>% arrange(rep_person_id, hosp_admit_dt) %>% setDT()

# no longer has bar seps
# MERGE COLUMNS
# merge with space
# for (key in keys) set(myhosp,which(is.na(myhosp[[key]])),key,"")
#fwrite(myhosp,"/infodev1/rep/projects/jason/myhosp_before_merge.csv")
#
# myhosp<- myhosp %>% unite("all_dx_codes",keys, sep=" ") %>% setDT()
# splitted<-strsplit(myhosp$all_dx_codes,"\\s+")
# # split with space, replace with bar
# pasted<-lapply(splitted,function(x) paste(x,collapse='|'))
# # mutate
# myhosp<- myhosp %>% mutate(dx_codes=pasted) %>% setDT()
# myhosp<- myhosp %>% select(-all_dx_codes,-hosp_dx_code_type) %>% setDT()

# # for the two labels, remove all below 1000 cases.
# myhosp<-myhosp %>% group_by(hosp_disch_disp) %>% mutate(n=n()) %>% setDT()
# myhosp<-myhosp %>% mutate(hosp_disch_disp=if_else(n<1000,"XXX",hosp_disch_disp)) %>% select(-n) %>% setDT()
#
# myhosp<-myhosp %>% group_by(hosp_adm_source) %>% mutate(n=n()) %>% setDT()
# myhosp<-myhosp %>% mutate(hosp_adm_source=if_else(n<1000,"XXX",hosp_adm_source)) %>% select(-n) %>% setDT()
#
# myhosp<-myhosp %>% mutate(is_in_patient=hosp_inout_code=="I") %>% select(-hosp_inout_code) %>% setDT()
# myhosp<-myhosp %>% arrange(rep_person_id, hosp_admit_dt) %>% setDT()
# myhosp<- myhosp %>% mutate(dx_codes=unlist(dx_codes)) %>% setDT()
myhosp2[hosp_disch_dt=="1800-01-01"]$hosp_disch_dt<-NA
fwrite(myhosp2,"/infodev1/rep/projects/jason/new/myhosp.csv")



######## SURGERIES
surg<-fread("/infodev1/rep/data/surgeries.dat")
mysurg<-surg%>%select(rep_person_id,px_date,px_codetype,px_code) %>%setDT()
# I am going to conver all I9 codes to ICD10 codes here. For table: https://www.cms.gov/medicare/coding/ICD10/2014-ICD-10-PCS.html
pcs<-fread("data/gem_i9pcs.txt")
colnames(pcs)<-c("i9","i10","flag")
pcs<- pcs %>% distinct(i9,.keep_all=T)
# fixed a bug here for multiple matches
# mysurg <- mysurg %>% left_join(pcs,by=c("px_code"="i9")) %>% setDT()
# no dot

mysurgi9 <- mysurg %>% filter(px_codetype=="I9") %>% setDT()
mysurgi9 <- mysurgi9 %>% separate(px_code,c("first","second"),remove=FALSE) %>%  setDT()
mysurgi9 <- mysurgi9 %>% replace_na(list(first="",second="")) %>% setDT()
mysurgi9 <- mysurgi9 %>% unite("nodot",c("first","second"),sep="") %>% setDT()
mysurgi9 <- mysurgi9 %>% mutate(nodot=as.integer(nodot)) %>% setDT()

all_i9 <- mysurgi9 %>% distinct(nodot) %>% setDT()



library(foreach)
library(parallel)
cl<-parallel::makeForkCluster(32)
doParallel::registerDoParallel(cl)
ret2<-foreach(i=1:nrow(all_i9), .combine='rbind') %dopar%{
    pcs%>% right_join(all_i9[i],by=c(i9="nodot"))}
stopCluster(cl)

ret2 <- ret2 %>% select(-flag)%>% setDT()
second_run <- ret2[is.na(i10)]
# after a few samples, I think the second run consists of two types: those that missed a 0 at the end, and those that
# are actually icd10 codes.
# first we append zero and search again
second_run_type_A<-second_run %>% mutate(i9=paste(i9,"0",sep="")) %>% select(-i10) %>% setDT()

cl<-parallel::makeForkCluster(8)
doParallel::registerDoParallel(cl)
second_run_type_A_ret<-foreach(i=1:nrow(second_run_type_A), .combine='rbind') %dopar%{
    gem%>% right_join(second_run_type_A[i],by=c(i9="i9"))}
stopCluster(cl)

second_run_type_A_ret <- second_run_type_A_ret %>% setDT()
second_run_type_A_ret <- second_run_type_A_ret %>% mutate(i9=second_run$i9) %>% setDT()
# 1230 rows found their matches
second_run_type_A_ret <-second_run_type_A_ret[!is.na(i10)]



second_run_type_B<- second_run %>% filter(!i9 %in% second_run_type_A_ret$i9) %>% setDT()
# many of them are i9 codes that are not specific enough to be converted. These codes are not touched.
# I am looking for exact matches of icd 10 codes, or else they will be considered no match.
second_run_type_B <- second_run_type_B %>% filter(i9 %in% gem$i10) %>% setDT()
second_run_type_B_ret <- second_run_type_B %>% mutate(i10=i9) %>% setDT()

second_run_type_C <- second_run %>% filter(!i9 %in% second_run_type_A_ret) %>% filter(! i9 %in% second_run_type_B_ret) %>% setDT()
second_run_type_C <-second_run_type_C %>% mutate(i9=paste(i9,"1",sep="")) %>% select(-i10) %>% setDT()

cl<-parallel::makeForkCluster(8)
doParallel::registerDoParallel(cl)
second_run_type_C_ret<-foreach(i=1:nrow(second_run_type_C), .combine='rbind') %dopar%{
    gem%>% right_join(second_run_type_C[i],by=c(i9="i9"))}
stopCluster(cl)

second_run_type_C_ret <- second_run_type_C_ret %>% setDT()
second_run_type_C_ret <- second_run_type_C_ret %>% mutate(i9=second_run$i9) %>% setDT()
second_run_type_C <- second_run %>% filter(!i9 %in% second_run_type_A_ret) %>% filter(! i9 %in% second_run_type_B_ret) %>% setDT()
second_run_type_C_ret <- second_run_type_C_ret %>% mutate(i9=second_run_type_C$i9) %>% setDT()


# 1230 rows found their matches
second_run_type_C_ret<-second_run_type_C_ret[!is.na(i10)]

ret2 <- ret2 %>% filter(!is.na(i10)) %>% setDT()
ret2 <- rbind(ret2, second_run_type_A_ret, second_run_type_B_ret, second_run_type_C_ret)
ret2 <- ret2 %>% filter(!is.na(i10)) %>% filter(i10!="NoDx")%>% setDT()


mysurgi9 <- mysurgi9 %>% left_join(ret2, by=c("nodot"="i9")) %>% setDT()



mysurgi9 <- mysurgi9 %>% select(-nodot, -px_code) %>% setDT()
mysurgi10 <-mysurg %>% filter(px_codetype=="I10") %>% mutate(i10=px_code) %>% select(-px_code) %>% setDT()

mysurg<-rbind(mysurgi9,mysurgi10) %>% setDT()
mysurg <- mysurg %>% mutate(px_date=mdy(px_date)) %>% setDT()
mysurg <- mysurg[!is.na(i10)] %>% mutate(rep_person_id=as.integer(rep_person_id)) %>% setDT()


# no tail elimination until we enter the post stage.
# # ICD9 is much better.
# # we eliminate tail.
# # The algorithm is simple. We see the count for each 4 digit code, and if the count is fewer than 2000, then we aggregate them to a 3 digit code
# # I can do it because ICD code is structured
# # 2000 and 4 are arbitrary decisions
# mysurg <- mysurg %>% group_by(px_code) %>% mutate(n=n()) %>% setDT()
# mysurg <- mysurg %>% mutate(other=n<2000) %>% setDT()
# mysurg <- mysurg %>% mutate(collapsed_px_code=if_else(other==T,as.integer(px_code%/%10),px_code)) %>% setDT()
# # I think it's worth it. The dimension has been collapsed to under 1000, compared to 20000.
# mysurg <- mysurg%>% select(-n,-other)
# mysurg <- mysurg %>% arrange(rep_person_id,px_date) %>% setDT()
# #fwrite(mys

mysurg <- mysurg %>% arrange(rep_person_id,px_date) %>% setDT()

fwrite(mysurg, "/infodev1/rep/projects/jason/new/mysurg.csv")


#################

# this is the post.R in death/post

hos<-fread('/infodev1/rep/projects/jason/new/myhosp.csv')
# all hos rows have admit dt
disch_hos<-hos[hosp_disch_dt!=""] %>% select(-hosp_admit_dt)
admit_hos<-hos %>% select(-hosp_disch_dt) %>% setDT()

fwrite(admit_hos,'/infodev1/rep/projects/jason/new/admit_hos.csv')
fwrite(disch_hos,'/infodev1/rep/projects/jason/new/disch_hos.csv')


####################

# This script makes a fake index so that multiindexing can be used in pandas.

require(lubridate)
require(data.table)
require(dplyr)
require(foreach)
require(parallel)

# find the first and the last date of all demo

death<-fread('/infodev1/rep/projects/jason/new/deathtargets.csv')
dia<-fread('/infodev1/rep/projects/jason/new/mydia_before_bar.csv')
ahos<-fread('/infodev1/rep/projects/jason/new/admit_hos.csv')
dhos<-fread('/infodev1/rep/projects/jason/new/disch_hos.csv')
surg<-fread("/infodev1/rep/projects/jason/new/mysurg.csv")

dfs=list("death","dia","ahos","dhos","surg")

cl<-parallel::makeForkCluster(6)
doParallel::registerDoParallel(cl)

# passing a list of strings has much higher performance than passing a list of datatables.

foreach(dfn=dfs) %dopar%{
    df<-get(dfn)
    df<- df %>% group_by(rep_person_id) %>% mutate(id=row_number())
    fwrite(df,paste('/infodev1/rep/projects/jason/new/multi',dfn,'.csv',sep=''))
    1
}
stopCluster(cl)


##########################
# add natural death causes records in cod dataset
require(lubridate)
mydeath<-fread('/infodev1/rep/projects/jason/new/multideath.csv')
demo<-fread('/infodev1/rep/data/demographics.dat')

dead<-demo[death_date!=""]
natural<-dead[!rep_person_id %in% mydeath$rep_person_id]
natural<-natural %>% select(rep_person_id,death_date) %>% setDT()
natural <- natural %>% mutate(id=1, code=0, underlying=F, death_date=mdy(death_date)) %>% setDT()
mydeath<- mydeath %>% mutate(death_date=ymd(death_date)) %>% setDT()
newdeath<-rbind(mydeath,natural)
newdeath<-newdeath%>% arrange(rep_person_id, id) %>% setDT()
newdeath<-newdeath[rep_person_id %in% demo$rep_person_id]

fwrite(newdeath,"/infodev1/rep/projects/jason/new/newdeath.csv")