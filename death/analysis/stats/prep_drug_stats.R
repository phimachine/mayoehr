# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 5/19/2019


require(data.table)
require(dplyr)
require(fuzzyjoin)

death<-fread('/infodev1/rep/projects/jason/new/deathtargets.csv')
dia<-fread('/infodev1/rep/projects/jason/new/mydia_before_bar.csv')
ahos<-fread('/infodev1/rep/projects/jason/new/admit_hos.csv')
dhos<-fread('/infodev1/rep/projects/jason/new/disch_hos.csv')

drug_cod<-fread('../../../drug/drugcod2.csv')
drug_usages<-fread('../../../drug/drugs.csv')

drug_cod<-drug_cod%>% mutate(icd10=paste("^",icd10, sep = "")) %>% setDT()

dddia<-dia%>% regex_left_join(drug_usages,by=c(nodot="icd10")) %>% filter(!is.na(icd10)) %>% setDT()
ddahos<-ahos %>% regex_left_join(drug_usages,by=c(nodot="icd10")) %>% filter(!is.na(icd10))  %>% setDT()
dddhos<-dhos %>% regex_left_join(drug_usages,by=c(nodot="icd10")) %>% filter(!is.na(icd10))  %>% setDT()
dddeath<-death %>% regex_left_join(drug_cod, by=c(code="icd10")) %>% filter(!is.na(icd10)) %>% setDT()

dia_count<-dddia %>% select(nodot, icd10) %>% setDT()
ahos_count<-ddahos %>% select(nodot, icd10) %>% setDT()
dhos_count<-dddhos %>% select(nodot, icd10) %>% setDT()


death_count<-dddeath %>% select(code, icd10) %>% setDT()
input_count<-rbind(dia_count,ahos_count, dhos_count)

fwrite(death_count, "drug/drug_death.csv")
fwrite(input_count, "drug/drug_input.csv")


ddia<-dddia %>% distinct(rep_person_id) %>% setDT()
dahos<-ddahos %>% distinct(rep_person_id) %>% setDT()
ddhos<-dddhos %>% distinct(rep_person_id) %>% setDT()
ddeath<-dddeath %>% distinct(rep_person_id) %>% setDT()

users<-rbind(ddia,dahos,ddhos)
# 5847
users<- users %>% distinct(rep_person_id) %>% setDT()

# 5847 users of drugs,  1465 deaths of drugs, 164 cases of intersections.
intersect(users, ddeath)