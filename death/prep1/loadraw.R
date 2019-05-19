# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 5/18/2019

require(data.table)
require(dplyr)

demo<-fread('/infodev1/rep/data/demographics.dat')
cod<-fread('/infodev1/rep/data/cause_of_death.csv')
dia<-fread('/infodev1/rep/data/diagnosis.csv')
hosp<-fread('/infodev1/rep/data/hospitalizations.dat')
serv<-fread('/infodev1/rep/data/services.dat')
vitals<-fread('/infodev1/rep/data/vitals.dat')
labs<-fread("/infodev1/rep/data/labs.dat",fill=TRUE)
pres<-fread('/infodev1/rep/data/prescriptions.csv')
surg<-fread("/infodev1/rep/data/surgeries.dat")
