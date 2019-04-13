# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 3/29/2019

require(data.table)
require(dplyr)
require(stringr)
require(ggplot2)
require(plotly)

setwd("D:\\Git\\mayoehr\\death\\analysis\\masscurve")
csvs<-list.files(path='../csvs')

curve<-function(colnames, dnconly=FALSE){
  # accepts colnames tcod, ttoe, ttt, tsen, tspe, troc, vcod, vtoe, vtt, vsen, vspe, vroc
  g<-ggplot()
  
  if (length(colnames)>1){
    multiple<-TRUE
  }else{
    multiple<-F
  }
  
  alldts<-list()
  
  if (multiple){
    metric<-substr(colnames[1],2,nchar(colnames[1]))
  }
  
  for (colname in colnames){
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
    
    if (multiple){
      # if there are multiple columns being selected
      # then I will change the list names 
      
      if (substr(colname[1],1,1)=="v"){
        prestr<-"valid"
      }else{
        prestr<-"train"
      }
      
      for (model_name in model_names){
        newdt<-dts[[model_name]]
        newdt<-newdt %>% mutate(run=prestr) %>% 
          mutate(!!metric:=!!as.name(colname)) %>% 
          select(-!!colname) %>% setDT()
        alldts[[paste(prestr, model_name, sep="")]]<-newdt
      }
    }else{
      alldts<-dts
    }
    
  }
  
  all_exps<-bind_rows(alldts)
  
  if (dnconly){
    all_exps<-all_exps %>% filter(grepl("DNC", experiment)) %>% setDT()
  }
  if (multiple){
    g<-g+geom_line(data=all_exps, aes_string(x="epoch", y=metric, color="experiment", group="run"))
  }else{
    g<-g+geom_line(data=all_exps, aes_string(x="epoch", y=colnames, color="experiment"))
  }
  
  
  # plot the plots
  ggplotly(g,width=900)
}

# curve(c("vroc","troc"),dnconly = TRUE)
# curve("vroc")
curve(c("vtt","ttt"),dnconly = TRUE)
