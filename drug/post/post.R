# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 2/26/2019


####################
# find the first and the last date of all demo

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
