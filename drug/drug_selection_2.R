# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 2/7/2019

# now that bar separated notation has been removed, it's possible to do everything in R. I suspect it will be faster
# with c++ backend.

# we are still looking at diag, hos, and death.

require(data.table)
require(dplyr)
require(fuzzyjoin)

death<-fread('/infodev1/rep/projects/jason/new/deathtargets.csv')
dia<-fread('/infodev1/rep/projects/jason/new/mydia_before_bar.csv')
ahos<-fread('/infodev1/rep/projects/jason/new/admit_hos.csv')
dhos<-fread('/infodev1/rep/projects/jason/new/disch_hos.csv')

drug_cod<-fread('drugcod2.csv')
drug_usages<-fread('drugs.csv')

drug_cod<-drug_cod%>% mutate(icd10=paste("^",icd10, sep = "")) %>% setDT()

ddia<-dia%>% regex_left_join(drug_usages,by=c(nodot="icd10")) %>% filter(!is.na(icd10)) %>% distinct(rep_person_id) %>% setDT()
dahos<-ahos %>% regex_left_join(drug_usages,by=c(nodot="icd10")) %>% filter(!is.na(icd10)) %>% distinct(rep_person_id) %>% setDT()
ddhos<-dhos %>% regex_left_join(drug_usages,by=c(nodot="icd10")) %>% filter(!is.na(icd10)) %>% distinct(rep_person_id) %>% setDT()

ddeath<-death %>% regex_left_join(drug_cod, by=c(code="icd10")) %>% filter(!is.na(icd10)) %>% distinct(rep_person_id) %>% setDT()


users<-rbind(ddia,dahos,ddhos)
users<- users %>% distinct(rep_person_id) %>% setDT()

# 5847 users of drugs,  1465 deaths of drugs, 164 cases of intersections.
intersect(users, ddeath)

# I want to examine the records of diagnosis, for example, to see if the specific drugs were mentioned.

rid_test<-ddia[sample(nrow(ddia),1)]
test<-dia %>% filter(rep_person_id==rid_test[1,rep_person_id])
test<- test %>% regex_left_join(drug_usages, by=c(nodot="icd10")) %>% filter(!is.na(icd10)) %>% setDT()
test

# let's look at deaths too
rid_test<-ddeath[sample(nrow(ddeath),1)]
test<-death %>% filter(rep_person_id==rid_test[1,rep_person_id])
test<- test %>% regex_left_join(drug_cod, by=c(code="icd10")) %>% filter(!is.na(icd10)) %>% setDT()
test

count_death<-death %>% regex_left_join(drug_cod, by=c(code="icd10")) %>% filter(!is.na(icd10)) %>% setDT()
count_death<-count_death %>% group_by(code) %>% mutate(n=n()) %>% distinct(code, n)%>%setDT()
count_death<-count_death %>% arrange(n) %>% setDT()