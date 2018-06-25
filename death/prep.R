# okay, let's do it again.
# this time, we will adhere to a month of precision, because the date is easy to parse

require(lubridate)
require(data.table)
require(dplyr)
require(doParallel)

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
main<-setDT(main%>%filter(code_type=="ICD10"))
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

#####
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

#####
hosp<-fread('/infodev1/rep/data/hospitalizations.dat')
myhosp<-hosp%>%select(rep_person_id, hosp_admit_dt,hosp_disch_dt,hosp_inout_code,hosp_adm_source,hosp_disch_disp, hosp_primary_dx,starts_with("hosp_secondary_dx"))
# no dirty data found, this is a carefully curated dataset.
# all of them seem to be ICD9 or ICD10
myhosp<-myhosp%>%mutate(hosp_admit_dt=mdy(hosp_admit_dt),hosp_disch_dt=mdy(hosp_disch_dt)) %>% setDT()
myhosp<-myhosp%>%arrange(rep_person_id,hosp_admit_dt)%>%setDT()
# myhosp has all the diagnosis codes expanded as dimensions
fwrite(myhosp,"/infodev1/rep/projects/jason/myhosp.csv")

#####
# from this point on, everything else will be repetitive
labs<-fread("/infodev1/rep/data/labs.dat",fill=TRUE) 
mylabs<-labs%>%select(rep_person_id, lab_date, lab_src_code, lab_loinc_code, lab_result, lab_range, lab_units, lab_abn_flag)%>%setDT()
mylabs<-mylabs%>%mutate(lab_date=substr(lab_date,1,9))%>%setDT()
mylabs<-mylabs%>%mutate(lab_date=dmy(lab_date))%>%setDT()
mylabs<-mylabs%>%select(-lab_src_code)%>%setDT()
# I am going to normalize lab_range, lab_results and lab_units very naively. I believe this naive normalization method would work better than feeding it directly in.
# I assume that all lab results are normal distribution. I assumet that all lab_ranges are intervals based on sigmas.
# normalized_measure=(lab_result-up_or_lower_bound)/lab_range_interval_size
# some other lab measures are treated as binary,

## this code stucks at one core
#lab_length<-nrow(mylabs)
#foreach(i=mylabs$lab_range,.combine='c')%dopar%{
#    strsplit(i,"-")
#}
## this code works, but somehow it's very slow.
cl<-makeCluster(8)
parLapply(cl,mylabs$lab_range,function(range){
    strsplit(range,"-")
})
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
splitted<-labs %>% separate(lab_range, c("A","B","C","D"),sep='-',remove=FALSE) %>% setDT()

#
