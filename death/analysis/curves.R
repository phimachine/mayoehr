# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 3/14/2019

require(data.table)
require(dplyr)

curve<-fread("curve.csv")
# I am mainly interested in the direction of troc and vroc
curve<-curve%>% mutate(epoch=seq.int(nrow(curve)))

require(ggplot2)
ggplot(data=curve)+
  geom_line(aes(x=epoch,y=ttt, color="training total"))+
  geom_line(aes(x=epoch,y=vtt, color="validation total"))

ggsave("loss.png")


ggplot(data=curve)+
  geom_line(aes(x=epoch,y=troc, color="training roc"))+
  geom_line(aes(x=epoch,y=vroc, color="validation roc"))

ggsave("roc.png)f"