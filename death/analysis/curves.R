# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 3/14/2019

require(data.table)
require(dplyr)
require(rstudioapi)
current_path <- getActiveDocumentContext()$path 
setwd(dirname(current_path ))

# bs=256
adnc<-fread("adnc.csv")
# I am mainly interested in the direction of troc and vroc
adnc<-adnc%>% mutate(epoch=seq.int(nrow(adnc)))

# bs=64
dnc<-fread("dnc_standard_param.csv")
dnc<-dnc%>% mutate(epoch=seq.int(nrow(dnc)))


require(ggplot2)
ggplot()+
  geom_line(aes(x=adnc$epoch,y=adnc$tcod, color="training", linetype ="adnc"))+
  geom_line(aes(x=adnc$epoch,y=adnc$vcod, color="validation", linetype ="adnc"))+
  geom_line(aes(x=dnc$epoch, y=dnc$tcod, color="training", linetype ="dnc"))+
  geom_line(aes(x=dnc$epoch,y=dnc$vcod, color="validation", linetype ="dnc"))+
  ggtitle("Cause of death loss")+
  xlab("Number of epochs")+
  ylab("Average binary cross entropy loss per label")
  

ggsave("loss.png")


ggplot()+
  geom_line(aes(x=dnc$epoch, y=dnc$troc, color="training", linetype= "dnc"))+
  geom_line(aes(x=dnc$epoch,y=dnc$vroc, color="validation", linetype= "dnc"))+
  geom_line(aes(x=adnc$epoch, y=adnc$troc, color="training", linetype="adnc"))+
  geom_line(aes(x=adnc$epoch,y=adnc$vroc, color="validation", linetype="adnc"))+
  xlab("Number of epochs")+
  ylab("Average receiver operating characteristic per label")

ggsave("roc.png")
