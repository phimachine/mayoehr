# This script makes a fake index so that multiindexing can be used in pandas.

require(lubridate)
require(data.table)
require(dplyr)
require(foreach)
require(parallel)

####################
# find the first and the last date of all demo

death<-fread('/infodev1/rep/projects/jason/deathtargets.csv')
demo<-fread('/infodev1/rep/projects/jason/demo.csv')
dia<-fread('/infodev1/rep/projects/jason/mydia.csv')
ahos<-fread('/infodev1/rep/projects/jason/admit_hos.csv')
dhos<-fread('/infodev1/rep/projects/jason/disch_hos.csv')
lab<-fread('/infodev1/rep/projects/jason/mylabs.csv')
pres<-fread('/infodev1/rep/projects/jason/mypres.csv')
serv<-fread("/infodev1/rep/projects/jason/myserv.csv")
surg<-fread("/infodev1/rep/projects/jason/newsurg.csv")
vital<-fread("/infodev1/rep/projects/jason/myvitals.csv")

dfs=list("death","dia","ahos","dhos","lab","pres","serv","surg","vital")

cl<-parallel::makeForkCluster(6)
doParallel::registerDoParallel(cl)

# passing a list of strings has much higher performance than passing a list of datatables.

foreach(dfn=dfs) %dopar%{
    df<-get(dfn)
    df<- df %>% group_by(rep_person_id) %>% mutate(id=row_number())
    fwrite(df,paste('/infodev1/rep/projects/jason/multi',dfn,'.csv',sep=''))
    1
}



