require(lubridate)

require(data.table)
require(dplyr)
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
cl<-parallel::makeForkCluster(48)
doParallel::registerDoParallel(cl)


ret<-foreach(i=demo$rep_person_id,.combine='rbind') %dopar% {
    mark_dates(i)
}


