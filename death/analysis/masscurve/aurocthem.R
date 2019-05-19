# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 5/18/2019

# to collect AUROC, you need to
# collect test set sensitivity and specificity on all models with /death/final/finaltest

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

