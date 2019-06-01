# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 5/19/2019

# only the code frequency is interesting. Nothing else matters.

require(data.table)
require(ggplot2)
require(stringr)
require(dplyr)

igstats_dir<-"igstats"

csvs<-list.files(igstats_dir)
df_col<-list()
# one code per df, known
for (csv in csvs){
    splitted<-str_split(csv,"_")[[1]]
    dfnn<-splitted[1]
    colnn<-paste(splitted[2:(length(splitted)-1)],collapse="_")
    df_col[dfnn]=colnn
}


for (dfn in names(df_col)){
  df_col_<-paste(dfn, df_col[[dfn]],sep="_")
  count_csv<-paste0(df_col_, "_count.csv")
  dict_csv<-paste0(df_col_, "_dict.csv")
  sort_csv<-paste0(df_col_, "_sort.csv")
  
  count_df<-fread(file.path(igstats_dir,count_csv)) %>% select(-V1) %>% setDT()
  dict_df<-fread(file.path(igstats_dir,dict_csv))%>% select(-V1) %>% setDT()
  sort_df<-fread(file.path(igstats_dir,sort_csv))%>% select(-V1) %>% setDT()
  
  combined<- dict_df %>% left_join(count_df) %>% setDT()
  assign(paste(df_col_, "df",sep="_"), combined, envir=.GlobalEnv)
}

