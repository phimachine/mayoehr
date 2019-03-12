
code_dist <- read.csv("C:/Users/JasonHu/code_dist")
View(code_dist)
library(ggplot2)
library(data.table)
require(dplyr)
other_code_dist<-data.table(code_dist)
small_occ<-code_dist%>% filter(n<10)
other<-sum(small_occ$n)
other_code_dist <- other_code_dist %>% filter(n>=10)
other_code_dist<-rbind(other_code_dist,list("other",other))
other_code_dist <- other_code_dist %>% arrange(desc(n)) %>% setDT()
bp<-ggplot(code_dist,aes(x=reorder(factor(code),-n),y=n,fill=factor(code)))+
  geom_bar(width=1,stat="identity")+
  xlab("ICD-9 codes")+
  ylab("count")+
  theme(axis.text.x = element_blank())+
  guides(fill=guide_legend(title="ICD-9 codes"))
bp
ggsave("C:\\Users\\JasonHu\\Desktop\\code_dist.png")