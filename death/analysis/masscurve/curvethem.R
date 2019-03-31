# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 3/29/2019

require(data.table)
require(dplyr)
require(stringr)
require(ggplot2)
require(plotly)

csvs<-list.files(path='../csvs')

curve<-function(colname){
  # accepts colnames tcod, ttoe, ttt, tsen, tspe, troc, vcod, vtoe, vtt, vsen, vspe, vroc
  
  
  # load the csv files
  dts<-list()
  model_names<-c()
  for (file in csvs){
    model_name<-str_split(file,"\\.")[[1]][1]
    model_names<-c(model_names,model_name)
    fpath<-paste("../csvs/",file, sep="")
    dt<-fread(fpath)
    newdt<-select(dt,colname,"epoch") %>% setDT()
    dts[[model_name]]<-newdt
  }
  
  # process the datasets, sometimes there are a few lines for a single epoch.
  for (model_name in model_names){
    dt<-dts[[model_name]]
    print_freq<-nrow(dt[epoch==0])
    nr<-nrow(dt)
    added<-(1/print_freq * (seq(1,nr)+print_freq-1) %% print_freq)
    dt<-dt %>% mutate(epoch=epoch+added) %>% setDT()
    dts[[model_name]]<-dt
  }

  # combine all data sets to row observation standards
  # 
  for (model_name in model_names){
    dt<-dts[[model_name]]
    dt<-dt %>% mutate(experiment=model_name) %>% setDT()
    dts[[model_name]]<-dt
  }

  all_exps<-bind_rows(dts)
  
  # plot the plots
  g<-ggplot()+
    geom_line(data=all_exps, aes_string(x="epoch", y=colname, color="experiment"))
  ggplotly(g)
}
curve("vtt")

