# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 3/29/2019

require(data.table)
require(dplyr)
require(stringr)
require(ggplot2)
require(plotly)

wd<-"D:\\Git\\mayoehr\\death\\analysis\\masscurve\\"
csv_path<-file.path(wd,"csv")
setwd(wd)
csvs<-list.files(path='csv')

get_model_names<-function(csvs){
  models<-c()
  for (csv in csvs){
    models<-c(models,str_split(csv, "_")[[1]][1])
  }
  unique(models)
}

model_names<-get_model_names(csvs)

# 
curve<-function(model_names, traincolnames=NULL, validcolnames=NULL){
  # all column names will be plotted and they will have different line type

  # read train dfs
  if (!is.null(traincolnames)){
    colnames<-traincolnames
    postfix<-"_train.csv"
  }
  else{
    colnames<-validcolnames
    postfix<-"_valid.csv"
  }
  dts<-list()
  
  for (model_name in model_names){
    fpath<-file.path(csv_path, paste0(model_name,postfix))
    dt<-fread(fpath)
    dt<-dt %>% mutate(senspe=sen+spe) %>% setDT()
    newdt<-select(dt,colnames,"epoch") %>% setDT()
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
  for (model_name in model_names){
    dt<-dts[[model_name]]
    dt<-dt %>% mutate(experiment=model_name) %>% setDT()
    dts[[model_name]]<-dt
  }
  
  # combine all dfs
  alldts<-rbindlist(dts)
  # melt to plot together
  alldts<-melt(alldts, id=c("epoch","experiment"))
  
  if (length(colnames)>1){
    g<-ggplot()+
      geom_smooth(data=alldts, aes_string(x="epoch", y="value", color="experiment", linetype="variable"), span=0.2, alpha=0.3)
  }else{
    g<-ggplot()+
      geom_line(data=alldts, aes_string(x="epoch", y="value", color="experiment"), size=1)
  }
  g
}

# curve(c("vroc","troc"),dnconly = TRUE)
# curve("vroc")
exclude<-c("simple")
model_names<-model_names[!model_names %in% exclude]
softmax<-c("tranforwardsoftmax", "tranmixedattnsoftmax","tranmixedforwardsoftmax", "tranattnsoftmax","softmaxADNC","softmaxDNC")
model_names<-model_names[!model_names %in% curve]
# g<-curve(model_names,traincolnames=c("total"))
curve(model_names,validcolnames=c("senspe"))

curve(softmax, validcolnames=c("senspe"))
