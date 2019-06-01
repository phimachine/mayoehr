# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 5/19/2019

require(data.table)
require(ggplot2)
fpath<-"../toe_frequency.csv"
toe<-fread(fpath)

ggplot()+
  geom_bar(aes(toe$V2))+
  xlim(-2,100)