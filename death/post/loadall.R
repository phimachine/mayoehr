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


