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