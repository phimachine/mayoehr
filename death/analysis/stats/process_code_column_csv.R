# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 5/19/2019

# I want top 3 code frequency and count
# I want top 10 plots
# preprocessed dataset

require(data.table)
require(dplyr)
require(ggplot2)

wd<-"csvs"
code_cols<-list(list(dfn="death", coln="code"),
                list(dfn="dia", coln="nodot"),
                list(dfn="hos", coln="nodot"),
                list(dfn="lab", coln="lab_loinc_code"),
                list(dfn="pres", coln="med_ingr_rxnorm_code"),
                list(dfn="serv", coln="srv_px_code"),
                list(dfn="surg", coln="i10"))

bar_plot_n<-20

for (codecol in code_cols){
    dfn=codecol$dfn
    coln=codecol$coln
    fname<-paste(dfn,coln,"code.csv",sep="_")
    fpath<-file.path(wd, fname)
    codecount<-fread(fpath)
    total<-sum(codecount$count)
    print(dfn)
    print(coln)
    print(total)
    tp<-codecount %>% mutate(perc=count/total) %>% setDT()
    if (dfn=="death"){
        tp<-tp[code!=0]
    }
    print(head(tp,3))
    
    to_plot<-tp[1:bar_plot_n]
    to_plot[,eval(coln)]=factor(to_plot[[coln]], levels=to_plot[[coln]])
    
    g<-ggplot()+
        geom_bar(data=to_plot,aes_string(x=coln, y="count"), stat="identity")+
        coord_flip()+
        theme(text = element_text(size=16))
    assign(paste(dfn, coln,"g",sep="_"), g)
}

plotsdir<-"plots"

death_code_g+labs(x="ICD-10 CM code in pre-processed mortalities dataset", y="Count")
ggsave(file.path(plotsdir, "prepdeath.png"))

dia_nodot_g+labs(x="ICD-10 CM code in pre-processed diagnoses dataset", y="Count")
ggsave(file.path(plotsdir, "prepdiag.png"))

hos_nodot_g+labs(x="ICD-10 CM code in pre-processed hospitalizations dataset", y="Count")
ggsave(file.path(plotsdir, "prephosp.png"))

lab_lab_loinc_code_g+labs(x="LOINC code in pre-processed labs dataset", y="Count")
ggsave(file.path(plotsdir, "preplab.png"))

pres_med_ingr_rxnorm_code_g+labs(x="Ingredient RxNorm code in pre-processed prescriptions dataset", y="Count")
ggsave(file.path(plotsdir, "preppres.png"))

serv_srv_px_code_g+labs(x="HCP or CPT code in pre-processed services dataset", y="Count")
ggsave(file.path(plotsdir, "prepserv.png"))

surg_i10_g+labs(x="ICD-10 PCS code in pre-processed surgeries dataset", y="Count")
ggsave(file.path(plotsdir, "prepsurg.png"))

