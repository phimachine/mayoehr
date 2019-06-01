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

model_rename<-list()

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
      geom_smooth(data=alldts, aes_string(x="epoch", y="value", color="experiment", linetype="variable"), span=0.2, alpha=0.3)+
      theme(legend.position="bottom")+
      theme(text = element_text(size=20))
    
  }else{
    g<-ggplot()+
      geom_line(data=alldts, aes_string(x="epoch", y="value", color="experiment"), size=1)+
      theme(legend.position="bottom")+
      theme(text = element_text(size=20))
    
  }
  g
}

plot_examples<-function(){
  # curve(c("vroc","troc"),dnconly = TRUE)
  # curve("vroc")
  exclude<-c("simple")
  model_names<-model_names[!model_names %in% exclude]
  softmax<-c("tranforwardsoftmax", "tranmixedattnsoftmax","tranmixedforwardsoftmax", "tranattnsoftmax","softmaxADNC","softmaxDNC")
  model_names<-model_names[!model_names %in% softmax]
  # g<-curve(model_names,traincolnames=c("total"))
  curve(model_names,validcolnames=c("total"))
  curve(model_names,validcolnames=c("senspe"))
  
  curve(softmax,validcolnames=c("total"))
  curve(softmax,validcolnames=c("senspe"))
}

plot_them<-function(model_names, col, train=T){
  if (train){
    curve(model_names,traincolnames = c(col))
  }else{
    curve(model_names,validcolnames = c(col))
  }
}

# ablation
ablation_names<-c("DNC","priorDNC")
plot_them(ablation_names,"total",T)+labs(x="Epoch", y="Total trainning loss")
ggsave("plots/prior_ablation_train_loss.png")
plot_them(ablation_names,"total",F)+labs(x="Epoch", y="Total trainning loss")
ggsave("plots/prior_ablation_valid_loss.png")
plot_them(ablation_names,"senspe",T)+labs(x="Epoch", y="Sensitivity + specificity")
ggsave("plots/prior_ablation_train_senspe.png")
plot_them(ablation_names,"senspe",F)+labs(x="Epoch", y="Sensitivity + specificity")
ggsave("plots/prior_ablation_valid_senspe.png")

# bce
exclude<-c("simple","DNC")
bce_models<-model_names[!model_names %in% exclude]
softmax<-c("tranforwardsoftmax", "tranmixedattnsoftmax","tranmixedforwardsoftmax", "tranattnsoftmax","softmaxADNC","softmaxDNC")
bce_models<-bce_models[!bce_models %in% softmax]

# total
plot_them(bce_models,"total",T)+labs(x="Epoch", y="Training total loss")
ggsave("plots/bce_train_loss.png")
plot_them(bce_models,"total",F)+labs(x="Epoch", y="Validation total loss")
ggsave("plots/bce_valid_loss.png")

# senspe
plot_them(bce_models,"senspe",T)+labs(x="Epoch", y="Sensitivity + specificity")
ggsave("plots/bce_train_senspe.png")
plot_them(bce_models,"senspe",F)+labs(x="Epoch", y="Sensitivity + specificity")
ggsave("plots/bce_valid_senspe.png")

# toe
plot_them(bce_models,"toe",T)+labs(x="Epoch", y="Training time to event smoothed L1 loss")
ggsave("plots/bce_train_toe.png")
plot_them(bce_models,"toe",F)+labs(x="Epoch", y="Validation time to event smoothed L1 loss")
ggsave("plots/bce_valid_toe.png")

# cod
plot_them(bce_models,"cod",T)+labs(x="Epoch", y="Training binary cross entropy loss")
ggsave("plots/bce_train_cod.png")
plot_them(bce_models,"cod",F)+labs(x="Epoch", y="Validation binary cross entropy loss")
ggsave("plots/bce_valid_cod.png")


# softmax models
# total
plot_them(softmax,"total",T)+labs(x="Epoch", y="Training total loss")
ggsave("plots/softmax_train_loss.png")
plot_them(softmax,"total",F)+labs(x="Epoch", y="Validation total loss")
ggsave("plots/softmax_valid_loss.png")

# senspe
plot_them(softmax,"senspe",T)+labs(x="Epoch", y="Sensitivity + specificity")
ggsave("plots/softmax_train_senspe.png")
plot_them(softmax,"senspe",F)+labs(x="Epoch", y="Sensitivity + specificity")
ggsave("plots/softmax_valid_senspe.png")

# toe
plot_them(softmax,"toe",T)+labs(x="Epoch", y="Training time to event smoothed L1 loss")
ggsave("plots/softmax_train_toe.png")
plot_them(softmax,"toe",F)+labs(x="Epoch", y="Validation time to event smoothed L1 loss")
ggsave("plots/softmax_valid_toe.png")

# cod
plot_them(softmax,"cod",T)+labs(x="Epoch", y="Training binary cross entropy loss")
ggsave("plots/softmax_train_cod.png")
plot_them(softmax,"cod",F)+labs(x="Epoch", y="Validation binary cross entropy loss")
ggsave("plots/softmax_valid_cod.png")


# prior ablation
plot_them(c("DNC","priorDNC"),"total",T)+labs(x="Epoch", y="Training total loss")
ggsave("plots/ablation_train_loss.png")
plot_them(c("DNC","priorDNC"),"total",F)+labs(x="Epoch", y="Validation total loss")
ggsave("plots/ablation_valid_loss.png")
plot_them(c("DNC","priorDNC"),"senspe",T)+labs(x="Epoch", y="Training sensitivity + specificity")
ggsave("plots/ablation_train_senspe.png")
plot_them(c("DNC","priorDNC"),"senspe",F)+labs(x="Epoch", y="Validation sensitivity + specificity")
ggsave("plots/ablation_valid_senspe.png")