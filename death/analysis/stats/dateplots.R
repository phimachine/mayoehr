# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 5/19/2019

require(ggplot2)
require(data.table)
require(dplyr)
require(lubridate)


dates_dir<-"dates"
csvs<-list.files(dates_dir)


for (csv in csvs){
    names<-str_split(csv,"_")[[1]]
    dfn<-names[1]
    coln<-paste(names[2:(length(names)-1)],collapse="_")
    dfn_coln<-paste(dfn,coln,sep="_")
    df<-fread(file.path(dates_dir,csv))
    dates<-ymd(df[[coln]])
    g<-ggplot()+
      geom_histogram(aes(dates),bins=100)+
      theme(text = element_text(size=16))
    g
    ggsave(paste0("plots/",dfn,"_",coln,".png"))
}

# death_death_date_g+labs(x="Death dates", y="Count")
# ggsave("plots/death_date.png")
# demo_birth_date_g+labs(x="Birth dates", y="Count")
# ggsave("plots/birth_date.png")
# dia_dx_date_g+labs(x="Diagnosis dates", y="Count")
# ggsave("plots/diag_date.png")
# hos_hosp_disch_dt_g+labs(x="Hospitalization discharge dates", y="Count")
# ggsave("plots/disch_date.png")
# hos_hosp_admit_dt_g+labs(x="Hospitalization admission dates", y="Count")
# ggsave("plots/admit_date.png")
# lab_lab_date_g+labs(x="Lab dates", y="Count")
# ggsave("plots/lab_date.png")
# pres_MED_DATE_g+labs(x="Prescription dates", y="Count")
# ggsave("plots/pres_date.png")
# serv_SRV_DATE_g+labs(x="Services dates", y="Count")
# ggsave("plots/serv_date.png")
# surg_px_date_g+labs(x="Surgeries dates", y="Count")
# ggsave("plots/surg_date.png")
# vitals_VITAL_DATE_g+labs(x="Vitals dates", y="Count")
# ggsave("plots/vitals_date.png")
