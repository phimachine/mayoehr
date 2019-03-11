# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 3/10/2019

require(lubridate)
require(data.table)
require(dplyr)
require(foreach)
require(parallel)

demo<-fread('/infodev1/rep/projects/jason/demo.csv')
dia<-fread('/infodev1/rep/projects/jason/new/mydia_before_bar.csv')
hos<-fread('/infodev1/rep/projects/jason/new/myhosp.csv')
lab<-fread('/infodev1/rep/projects/jason/mylabs.csv')
pres<-fread('/infodev1/rep/projects/jason/mypres.csv')
serv<-fread("/infodev1/rep/projects/jason/myserv.csv")
surg<-fread("/infodev1/rep/projects/jason/new/mysurg.csv")
vitals<-fread("/infodev1/rep/projects/jason/myvitals.csv")
