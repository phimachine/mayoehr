# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 5/18/2019

require(lubridate)
require(data.table)
require(dplyr)
require(foreach)
require(parallel)
require(Hmisc)

demo<-fread('/infodev1/rep/projects/jason/demo.csv')
dia<-fread('/infodev1/rep/projects/jason/new/mydia_before_bar.csv')
hos<-fread('/infodev1/rep/projects/jason/new/myhosp.csv')
lab<-fread('/infodev1/rep/projects/jason/mylabs.csv')
pres<-fread('/infodev1/rep/projects/jason/mypres.csv')
serv<-fread("/infodev1/rep/projects/jason/myserv.csv")
surg<-fread("/infodev1/rep/projects/jason/new/mysurg.csv")
vitals<-fread("/infodev1/rep/projects/jason/myvitals.csv")
death<-fread("/infodev1/rep/projects/jason/newdeath.csv")

dfs<-c("demo", "dia", "hos", "lab", "pres", "serv", "surg", "vitals", "death")

# these functions do not run, it's as if the for loop is run asynchronously
# postfix<-"_prep.txt"
# for (dfn in dfs[1:4]){
#     df<-get(dfn)
#     sink(paste0(dfn, postfix))
#     describe(df)
#     sink()
# }
# sink()
#
# for (dfn in dfs[4:length(dfs)]){
#     df<-get(dfn)
#     sink(paste0(dfn, postfix))
#     describe(df)
#     sink()
# }

sink("demo_prep.txt")
describe(demo)

sink("dia_prep.txt")
describe(dia)

sink("hos_prep.txt")
describe(hos)

sink("lab_prep.txt")
lab <- lab[is.finite(bigger),]
lab <- lab[is.finite(smaller),]
describe(lab)

sink("pres_prep.txt")
describe(pres)

sink("serv_prep.txt")
describe(serv)

sink("surg_prep.txt")
describe(surg)

sink("vitals_prep.txt")
describe(vitals)

sink("death_prep.txt")
describe(death)
sink()

# code frequencies

code_cols<-list(list(dfn="death", coln="code"),
                list(dfn="dia", coln="nodot"),
                list(dfn="hos", coln="nodot"),
                list(dfn="lab", coln="lab_loinc_code"),
                list(dfn="pres", coln="med_ingr_rxnorm_code"),
                list(dfn="serv", coln="srv_px_code"),
                list(dfn="surg", coln="i10"))

for (codecol in code_cols){
    dfn=codecol$dfn
    coln=codecol$coln
    df=get(dfn)
    newdt<-df %>% group_by_(.dots=coln) %>% mutate(count=n()) %>% setDT()
    newdt<-newdt %>% distinct_(coln, "count") %>% arrange(desc(count)) %>% setDT()
    fwrite(newdt,paste(dfn,coln,"code.csv",sep="_"))
}

# date frequencies
date_cols<-list(list(dfn="death", coln="death_date"),
                list(dfn="demo", coln="birth_date"),
                list(dfn="dia", coln="dx_date"),
                list(dfn="hos", coln="hosp_admit_dt"),
                list(dfn="hos", coln="hosp_disch_dt"),
                list(dfn="lab", coln="lab_date"),
                list(dfn="pres", coln="MED_DATE"),
                list(dfn="serv", coln="SRV_DATE"),
                list(dfn="surg", coln="px_date"),
                list(dfn="vitals", coln="VITAL_DATE"))

for (datecol in date_cols){
    dfn=datecol$dfn
    coln=datecol$coln
    df=get(dfn)
    dates<-df %>% select_(.dots=coln) %>% setDT()
    if (nrow(dates)>100000){
        dates<-dates[sample(.N,100000)]
    }
    fwrite(dates,paste0("dates/",paste(dfn,coln,"date.csv",sep="_")))
}