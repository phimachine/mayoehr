# okay, let's do it again.
# this time, we will adhere to a month of precision, because the date is easy to parse

require(lubridate)
require(data.table)
require(dplyr)
require(doParallel)
require(tidyr)
require(fuzzyjoin)
require(xml2)
require(XML)

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
ingr_cui<-lapply(calls,function(x) xml_find_all(x,".//function[@level='1']//output/RXCUI")
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
fwrite(xml_text(inputs)%>%as.list(),'/infodev1/rep/projects/jason/parsed_inputs.csv')
hello<-lapply(medrxcui, function(x) { xml_text(x)[1] })
fwrite(hello %>% as.list,'/infodev1/rep/projects/jason/parsed_medrxcui.csv')
hello<-list()
hello<-lapply(1:length(inputs), function(x) { hello[[x]]<-xml_text(ingr_cui[[x]])  })
saveRDS(hello,file='/infodev1/rep/projects/jason/parsed_ingr_rxcui.rds')
chosen<-c()
chosen<-lapply(1:length(inputs), function (x) {chosen<-c(chosen,xml_text(chosenmedrx[[x]])[1])})
fwrite(chosen, '/infodev1/rep/projects/jason/parsed_first_queryed_medrxcui.csv')

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
fwrite(try,"/infodev1/rep/projects/jason/verbose_mypres.csv")
try_min <- try %>% select(-med_name,-med_generic,-rxnorm, -med_ingr_rxnorm_code) %>% setDT()
fwrite(try_min,'/infodev1/rep/projects/jason/min_mypres.csv')

# we need to create the bar comma file for mapping from med_rxnorm_code to ingr_rxnorm_code


###### SERVICES
# my intuition tells me that sevices will not beo too vital
# I will filter our the tail of the dataset to control input complexity.
serv<-fread('/infodev1/rep/data/services.dat')
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
