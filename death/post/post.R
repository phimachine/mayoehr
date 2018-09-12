require(lubridate)

require(data.table)
require(dplyr)


####################
# find the first and the last date of all demo

death<-fread('/infodev1/rep/projects/jason/deathtargets.csv')
demo<-fread('/infodev1/rep/projects/jason/demo.csv')
dia<-fread('/infodev1/rep/projects/jason/mydia.csv')
hos<-fread('/infodev1/rep/projects/jason/myhosp.csv')
lab<-fread('/infodev1/rep/projects/jason/mylabs.csv')
pres<-fread('/infodev1/rep/projects/jason/mypres.csv')
serv<-fread("/infodev1/rep/projects/jason/myserv.csv")
surg<-fread("/infodev1/rep/projects/jason/mysurg.csv")
vitals<-fread("/infodev1/rep/projects/jason/myvitals.csv")

dfs=list(dia,hos,lab,pres,serv,surg,vitals)
# find the earliest record and latest record, put them in demo

mark_dates<- function(rep_id){
    dates<-dia[rep_person_id==rep_id]$dx_date
    dates<-c(dates,hos[rep_person_id==rep_id]$hosp_admit_dt)
    dates<-c(dates,hos[rep_person_id==rep_id]$hosp_disch_dt)
    dates<-c(dates,lab[rep_person_id==rep_id]$lab_date)
    dates<-c(dates,pres[rep_person_id==rep_id]$MED_DATE)
    dates<-c(dates,serv[rep_person_id==rep_id]$SRV_DATE)
    dates<-c(dates,surg[rep_person_id==rep_id]$px_date)
    dates<-c(dates,vitals[rep_person_id==rep_id]$VITAL_DATE)
    dates<-ymd(dates)
    dates<-dates[!is.na(dates)]
    data.table(rep_person_id=rep_id,earliest=min(dates),latest=max(dates))
}

library(foreach)
library(parallel)

# not all 16 cores will be used.
cl<-parallel::makeForkCluster(4)
doParallel::registerDoParallel(cl)


ret<-foreach(i=demo$rep_person_id,.combine='rbind') %dopar% {
    mark_dates(i)
}

ret <- ret %>% arrange(rep_person_id) %>% setDT()
# our algorithm is way too advanced that it counts days
day(ret$earliest) <- 1
day(ret$latest) <- 1
ret <- ret %>% mutate(int=interval(earliest,latest) %/% months(1)) %>% setDT()


fwrite(ret,'/infodev1/rep/projects/jason/earla.csv')



#########################
# split hospitalization file since it has two dates? what the heck?
hos<-fread('/infodev1/rep/projects/jason/myhosp.csv')
# all hos rows have admit dt
disch_hos<-hos[hosp_disch_dt!=""] %>% select(-hosp_admit_dt)
admit_hos<-hos %>% select(-hosp_disch_dt) %>% setDT()

fwrite(admit_hos,'/infodev1/rep/projects/jason/admit_hos.csv')
fwrite(disch_hos,'/infodev1/rep/projects/jason/disch_hos.csv')


##########################
# add natural death causes records in cod dataset
require(lubridate)
mydeath<-fread('/infodev1/rep/projects/jason/multideath.csv')
demo<-fread('/infodev1/rep/data/demographics.dat')

dead<-demo[death_date!=""]
natural<-dead[!rep_person_id %in% mydeath$rep_person_id]
natural<-natural %>% select(rep_person_id,death_date) %>% setDT()
natural <- natural %>% mutate(id=1, code=0, underlying=F, death_date=mdy(death_date)) %>% setDT()
mydeath<- mydeath %>% mutate(death_date=ymd(death_date)) %>% setDT()
newdeath<-rbind(mydeath,natural)
newdeath<-newdeath%>% arrange(rep_person_id, id) %>% setDT()
fwrite(newdeath,"/infodev1/rep/projects/jason/newdeath.csv")


##### newdeath contains patients with no EHR records
newdeath<-newdeath[rep_person_id %in% demo$rep_person_id]
fwrite(newdeath,"/infodev1/rep/projects/jason/newdeath.csv")

####################

# This script makes a fake index so that multiindexing can be used in pandas.

require(lubridate)
require(data.table)
require(dplyr)
require(foreach)
require(parallel)

# find the first and the last date of all demo

death<-fread('/infodev1/rep/projects/jason/deathtargets.csv')
demo<-fread('/infodev1/rep/projects/jason/demo.csv')
dia<-fread('/infodev1/rep/projects/jason/mydia.csv')
ahos<-fread('/infodev1/rep/projects/jason/admit_hos.csv')
dhos<-fread('/infodev1/rep/projects/jason/disch_hos.csv')
lab<-fread('/infodev1/rep/projects/jason/mylabs.csv')
pres<-fread('/infodev1/rep/projects/jason/mypres.csv')
serv<-fread("/infodev1/rep/projects/jason/myserv.csv")
surg<-fread("/infodev1/rep/projects/jason/newsurg.csv")
vital<-fread("/infodev1/rep/projects/jason/myvitals.csv")

dfs=list("death","dia","ahos","dhos","lab","pres","serv","surg","vital")

cl<-parallel::makeForkCluster(6)
doParallel::registerDoParallel(cl)

# passing a list of strings has much higher performance than passing a list of datatables.

foreach(dfn=dfs) %dopar%{
    df<-get(dfn)
    df<- df %>% group_by(rep_person_id) %>% mutate(id=row_number())
    fwrite(df,paste('/infodev1/rep/projects/jason/multi',dfn,'.csv',sep=''))
    1
}
