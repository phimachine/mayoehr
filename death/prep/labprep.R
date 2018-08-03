# okay, let's do it again.
# this time, we will adhere to a month of precision, because the date is easy to parse

require(lubridate)
require(data.table)
require(dplyr)
require(doParallel)
require(tidyr)

##### LABS
labs<-fread("/infodev1/rep/data/labs.dat",fill=TRUE) 
mylabs<-labs%>%select(rep_person_id, lab_date, lab_src_code, lab_loinc_code, lab_result, lab_range, lab_units, lab_abn_flag)%>%setDT()
mylabs<-mylabs%>%mutate(lab_date=substr(lab_date,1,9))%>%setDT()
mylabs<-mylabs%>%mutate(lab_date=dmy(lab_date))%>%setDT()
mylabs<-mylabs%>%select(-lab_src_code)%>%setDT()
splitted<- mylabs %>% separate(lab_range, c("X","Y"), sep="to", remove=FALSE) %>% setDT()
splitted<- splitted %>% separate(X,c("A","B","C","D"), sep='-', remove=FALSE) %>% setDT()
splitted<-splitted[,A:=as.double(A)][,B:=as.double(B)][,C:=as.double(C)][,D:=as.double(D)][,Y:=as.double(Y)]
sum(is.na(splitted$A) && !is.na(splitted$D))
negative<-splitted[is.na(A)][!is.na(B)]
negative[,A:=-B]
negative[,B:=C]
negative[,C:=NA]
negative[!is.na(Y)][,"B"]<-negative[!is.na(Y)][,"Y"]
splitted[is.na(A)][!is.na(B)]<-negative
splitted[is.na(B)][!is.na(A)][,'A']<-NA
# now we have reached a point where extra information is clean.
# we want to see how abnormal a measurement is, then we will throw away all additional information and only retain the lab_delta and lab_abn_flag
splitted[,lab_result:=as.double(lab_result)]
# meaningful if smaller is negative
splitted_smaller<-splitted$lab_result-splitted$A
splitted_smaller<-replace_na(splitted_smaller,0)
splitted_smaller[splitted_smaller>0]<-0
splitted<-splitted %>% mutate(smaller=splitted_smaller) %>% setDT()
# meaningful if positive
splitted_bigger<-splitted$lab_result-splitted$B
splitted_bigger<-replace_na(splitted_bigger,0)
splitted_bigger[splitted_bigger<0]<-0
splitted<-splitted %>% mutate(bigger=splitted_bigger) %>% setDT()
# get range
splitted_range<-splitted$B - splitted$A
splitted_range<-replace_na(splitted_range,0)
# normalize
splitted<- splitted %>% mutate(range=splitted_range) %>% setDT()
splitted<- splitted %>% mutate(smaller=smaller/range, bigger=bigger/range) %>% setDT()
splitted[is.nan(smaller),'smaller']<-0
splitted[is.nan(bigger),'bigger']<-0
# we have finished normalization.
# for the last step, we can feed the abnormality flag to be a value, but I will not do it.
fwrite(splitted,"/infodev1/rep/projects/jason/mylabs.csv")

