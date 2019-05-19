# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 5/18/2019

# collect statistics of the raw data

require(data.table)
require(dplyr)
require(Hmisc)


sink("demo_raw.txt")
demo<-fread('/infodev1/rep/data/demographics.dat')
describe(demo)
rm(demo)

sink("dia_raw.txt")
dia<-fread('/infodev1/rep/data/diagnosis.csv')
describe(dia)
rm(dia)

sink("hos_raw.txt")
hosp<-fread('/infodev1/rep/data/hospitalizations.dat')
describe(hosp)
rm(hosp)

sink("lab_raw.txt")
labs<-fread("/infodev1/rep/data/labs.dat",fill=TRUE)
describe(labs)
rm(labs)

sink("pres_raw.txt")
pres<-fread('/infodev1/rep/data/prescriptions.csv')
describe(pres)
rm(pres)

sink("serv_raw.txt")
serv<-fread('/infodev1/rep/data/services.dat')
describe(serv)
rm(serv)

sink("surg_raw.txt")
surg<-fread("/infodev1/rep/data/surgeries.dat")
describe(surg)
rm(surg)

sink("vitals_raw.txt")
vitals<-fread('/infodev1/rep/data/vitals.dat')
describe(vitals)
rm(vitals)

sink("death_raw.txt")
cod<-fread('/infodev1/rep/data/cause_of_death.csv')
describe(cod)
rm(cod)
sink()

