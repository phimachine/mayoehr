# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 1/3/2019

require(data.table)
require(dplyr)
require(ggplot2)

hello <- hello %>% group_by(rep_person_id) %>% mutate(n=n()) %>% setDT()
hello <- hello %>% distinct(rep_person_id, n) %>% setDT()
hello <- hello %>% group_by(n) %>% mutate(nn=n()) %>% setDT()
hello <- hello %>% distinct(n,nn) %>% setDT()
ggplot(hello, aes(x=n,y=nn/sum(nn))) + geom_bar(stat='identity')
    +labs( y="Percentage of patients in the mortality dataset", x='Number of labels per patient')
    +scale_y_continuous(sec.axis=sec_axis(~.*ss, name="Count of patients"))