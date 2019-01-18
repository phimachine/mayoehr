# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 1/15/2019


require(data.table)
require(dplyr)
require(fuzzyjoin)

gem<-fread('/local2/tmp/pycharm_project_292/drug/2018_I10gem.txt')
drugs<-fread('/local2/tmp/pycharm_project_292/drug/drugs.csv')
colnames(gem)<-c("i10","i9","flags")
colnames(drugs)<-"name"
regex_result <- gem %>% regex_right_join(drugs, by=c(i10="name")) %>% setDT()
drugi9<-unique(regex_result[i9!="NoDx",i9])
fwrite(list(drugi9),"/infodev1/rep/projects/jason/drugsi9.csv")