# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 1/13/2019

require(data.table)
require(dplyr)
require(fuzzyjoin)


gem<-fread('/local2/tmp/pycharm_project_292/drug/2018_I10gem.txt')
colnames(gem)<-c("i10","i9","flags")
drugs<-fread('/local2/tmp/pycharm_project_292/drug/drugcod2.csv')

# match any beginning of the string
drugs <- drugs %>% mutate(name=paste('^',name,sep="")) %>% setDT()
regex_result <- gem %>% regex_right_join(drugs, by=c(i10="name")) %>% setDT()
# matched 739 rows of i9 codes that are drug related deaths
regex_result <- regex_result %>% filter(!is.na(i9))
drugi9<-unique(regex_result[!is.na(i9),i9])
fwrite(list(drugi9),"/infodev1/rep/projects/jason/drugcodi9.csv")
nd<-fread("/infodev1/rep/projects/jason/newdeath.csv")
drugi9<-data.table(drugi9)
drugi9<-drugi9 %>% mutate(drugi9=paste('^',drugi9,sep="")) %>% setDT()
drug_rep_id <- nd %>% regex_right_join(drugi9, by=c(code="drugi9")) %>% setDT()
# 1220 cases
drug_rep_id<-drug_rep_id[!is.na(rep_person_id), rep_person_id]
fwrite(list(drug_rep_id),"/infodev1/rep/projects/jason/drugrepid.csv")