require(data.table)
require(dplyr)
require(stringr)
require(ggplot2)
require(testit)
require(xtable)

options(warn=2)
# the wd should be /death/final
stats_dir <- "../unified/test_stats"
id<-"20"
experiments <- list.files(stats_dir)
new_experiments<-c()
for (experiment in experiments){
  new_experiment<-str_split(experiment,"_")[[1]]
  new_experiment<-paste(new_experiment[1:length(new_experiment)-1],collapse="_")
  new_experiments<-c(new_experiments,new_experiment)
}

get_auroc <- function(experiment) {
  wd <- file.path(stats_dir, paste(experiment,"stats", sep="_"))
  
  # first we collect the model names
  files <- list.files(wd)
  model_names <- c()
  for (file in files) {
    model_name <- str_split(file, "_")[[1]][1]
    if (!model_name %in% model_names) {
      model_names <- c(model_names, model_name)
    }
  }
  
  for (model in model_names) {
    conditional <-
      fread(file.path(wd, paste(model, "_conditional.csv", sep = "")))
    colnames(conditional)[1] <- "targetidx"
    
    tp <- fread(file.path(wd, paste(model, "_tp.csv", sep = "")))
    cname <- unlist(tp[1], use.names = F)
    cname[1] <- "threshold"
    colnames(tp) <- cname
    tp <- tp[2:nrow(tp)]
    tp[, "threshold"] <- tp[, "threshold"] / nrow(tp)
    tp <- setDT(tp)
    
    tn <- fread(file.path(wd, paste(model, "_tn.csv", sep = "")))
    cname <- unlist(tn[1], use.names = F)
    cname[1] <- "threshold"
    colnames(tn) <- cname
    tn <- tn[2:nrow(tn)]
    tn[, "threshold"] <- tn[, "threshold"] / nrow(tn)
    tn <- setDT(tn)
    
    # replace conditional zeros
    conditional <-
      conditional %>% mutate(cp = ifelse(cp == 0, 1e-5, cp)) %>% mutate(cn =
                                                                          ifelse(cn == 0, 1e-5, cn)) %>% setDT()
    
    # when probability > threshold, the prediction is True
    # get the sen/spe for each code, then average them.
    sen <- copy(tp)
    sen <- sen[, 2:ncol(sen)]
    for (i in seq_along(sen)) {
      sen[, (i) := sen[[i]] / conditional$cp[i]]
    }
    # take the average
    avg_sen <- rowMeans(sen)
    
    spe <- copy(tn)
    spe <- spe[, 2:ncol(spe)]
    for (i in seq_along(spe)) {
      spe[, (i) := spe[[i]] / conditional$cn[i]]
    }
    
    avg_spe <- rowMeans(spe)
    
    auroc <- data.table(sen = avg_sen, spe = avg_spe)
    auroc <-
      auroc %>% mutate(threshold = seq(
        0,
        by = 1 / nrow(auroc),
        length.out = nrow(auroc)
      )) %>% setDT()
    auroc <- bind_rows(auroc, list(
      sen = 0,
      spe = 1,
      threshold = 1
    ))
    
    # to calculate the area under, we integrate the surface with polar coordinate
    total_area <- 0
    for (i in seq(1, nrow(auroc) - 1)) {
      angle1 <- atan(auroc[i, spe] / auroc[i, sen])
      angle2 <- atan(auroc[i + 1, spe] / auroc[i + 1, sen])
      diff_angle <- angle2 - angle1
      assert(diff_angle >= 0)
      low_edge <- sqrt(auroc[i, spe] ^ 2 + auroc[i, sen] ^ 2)
      high_edge <- sqrt(auroc[i + 1, spe] ^ 2 + auroc[i + 1, sen] ^ 2)
      area <- high_edge * low_edge * sin(diff_angle) / 2
      total_area <- total_area + area
    }
    assign(paste("area", experiment, model, id, sep = "_"), total_area, envir = .GlobalEnv)
    print(paste(experiment, model, "auroc done"))
  }
}

plot_experiment <- function(experiment) {
  wd <- file.path(stats_dir, paste0(experiment,"_stats"))
  
  # first we collect the model names
  files <- list.files(wd)
  model_names <- c()
  for (file in files) {
    model_name <- str_split(file, "_")[[1]][1]
    if (!model_name %in% model_names) {
      model_names <- c(model_names, model_name)
    }
  }
  
  
  for (model in model_names) {
    conditional <-
      fread(file.path(wd, paste(model, "_conditional.csv", sep = "")))
    colnames(conditional)[1] <- "targetidx"
    
    tp <- fread(file.path(wd, paste(model, "_tp.csv", sep = "")))
    cname <- unlist(tp[1], use.names = F)
    cname[1] <- "threshold"
    colnames(tp) <- cname
    tp <- tp[2:nrow(tp)]
    tp[, "threshold"] <- tp[, "threshold"] / nrow(tp)
    tp <- setDT(tp)
    
    tn <- fread(file.path(wd, paste(model, "_tn.csv", sep = "")), fill=T)
    cname <- unlist(tn[1], use.names = F)
    cname[1] <- "threshold"
    colnames(tn) <- cname
    tn <- tn[2:nrow(tn)]
    tn[, "threshold"] <- tn[, "threshold"] / nrow(tn)
    tn <- setDT(tn)
    
    # replace conditional zeros
    conditional <-
      conditional %>% mutate(cp = ifelse(cp == 0, 1e-5, cp)) %>% mutate(cn =
                                                                          ifelse(cn == 0, 1e-5, cn)) %>% setDT()
    
    # when probability > threshold, the prediction is True
    # get the sen/spe for each code, then average them.
    sen <- copy(tp)
    sen <- sen[, 2:ncol(sen)]
    for (i in seq_along(sen)) {
      sen[, (i) := sen[[i]] / conditional$cp[i]]
    }
    # take the average
    avg_sen <- rowMeans(sen)
    
    spe <- copy(tn)
    spe <- spe[, 2:ncol(spe)]
    for (i in seq_along(spe)) {
      spe[, (i) := spe[[i]] / conditional$cn[i]]
    }
    
    avg_spe <- rowMeans(spe)
    
    auroc <- data.table(sen = avg_sen, spe = avg_spe)
    auroc <-
      auroc %>% mutate(threshold = seq(
        0,
        by = 1 / nrow(auroc),
        length.out = nrow(auroc)
      )) %>% setDT()
    auroc <- bind_rows(auroc, list(
      sen = 0,
      spe = 1,
      threshold = 1
    ))
    
    # AUROC plot
    g<-ggplot()+
      geom_line(aes(x = 1 - auroc$sen, y = auroc$spe), size = 2) +
      xlab("1-Sensitivity") +
      ylab("Specificity")
    experiment <- str_split(experiment, "_")[[1]][1]
    assign(paste("g", experiment, model, id,sep = "_"), g, envir = .GlobalEnv)
    print(paste(experiment, model, "plot done"))
  }
}

# plot_experiment("lstm")

for (experiment in new_experiments){
  plot_experiment(experiment)
  ggsave(file.path('auroc_plots/',paste0(experiment, "_auroc.png")))
  get_auroc(experiment)
}

# the wd should be /death/final
stats_dir <- "../unified/earlystopping_stats"
id<-"4"
experiments <- list.files(stats_dir)
new_experiments<-c()
for (experiment in experiments){
  new_experiment<-str_split(experiment,"_")[[1]]
  new_experiment<-paste(new_experiment[1:length(new_experiment)-1],collapse="_")
  new_experiments<-c(new_experiments,new_experiment)
}

for (experiment in new_experiments){
  plot_experiment(experiment)
  ggsave(file.path('auroc_plots/',paste0(experiment, "_auroc_4.png")))
  get_auroc(experiment)
}

# all aurocs calculated

compile_aurocs<-function(){
  
  aurocs<-data.table(model=character(), epoch=integer(), auroc=numeric())
  
  stats_dir <- "../unified/test_stats"
  experiments <- list.files(stats_dir)
  new_experiments<-c()
  for (experiment in experiments){
    new_experiment<-str_split(experiment,"_")[[1]]
    new_experiment<-paste(new_experiment[1:length(new_experiment)-1],collapse="_")
    new_experiments<-c(new_experiments,new_experiment)
  }
  
  id<-"20"
  for (experiment in new_experiments){
    wd <- file.path(stats_dir, paste0(experiment,"_stats"))
    # first we collect the model names
    files <- list.files(wd)
    model_names <- c()
    for (file in files) {
      model_name <- str_split(file, "_")[[1]][1]
      if (!model_name %in% model_names) {
        model_names <- c(model_names, model_name)
      }
    }
    for (model in model_names){
      au<-get(paste("area", experiment, model, id, sep = "_"), envir = .GlobalEnv)
      aurocs<-rbind(aurocs, list("model"=paste(experiment, model, sep="_"), "epoch"=id, "auroc"=au))
      
    }
  }
  
  
  stats_dir <- "../unified/earlystopping_stats"
  experiments <- list.files(stats_dir)
  new_experiments<-c()
  for (experiment in experiments){
    new_experiment<-str_split(experiment,"_")[[1]]
    new_experiment<-paste(new_experiment[1:length(new_experiment)-1],collapse="_")
    new_experiments<-c(new_experiments,new_experiment)
  }
  id<-"4"
  for (experiment in new_experiments){
    wd <- file.path(stats_dir, paste0(experiment,"_stats"))
    # first we collect the model names
    files <- list.files(wd)
    model_names <- c()
    for (file in files) {
      model_name <- str_split(file, "_")[[1]][1]
      if (!model_name %in% model_names) {
        model_names <- c(model_names, model_name)
      }
    }
    for (model in model_names){
      au<-get(paste("area", experiment, model, id, sep = "_"), envir = .GlobalEnv)
      aurocs<-rbind(aurocs, list("model"=paste(experiment, model, sep="_"), "epoch"=id, "auroc"=au))
    
    }
  }
  aurocs
}
aurocs<-compile_aurocs()
aurocs<-dcast(aurocs,model~epoch)
aurocs<-aurocs[model!="simple_simple"]
