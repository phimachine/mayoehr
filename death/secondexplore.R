# some people die more than once?

> length(unique(cod$rep_person_id))
[1] 18562
> nrow(cod)
[1] 90294

# asingle patient can have multiple entries, becuase there are multiple injury icd codes, we need to merge that.

# NA are integers and strings, but it's okay
> is.na(main2[1]$age_years)
[1] TRUE

> is.na(main2[1]$res_county)
[1] TRUE

# maybe it's not better to make new row dimensions than column dimensions
temp<-cod %>% group_by(rep_person_id)%>%mutate(n=n())
# given how many ICD codes a patient can have, 38, it's not possible to expand them into columns
max(temp$n)
head(sort(temp$n,decreasing=T),n=1000)

# I'm thinking about the feasibility of combining all these files.
# If I keep using joins, then the nrow is going to be exponential.
# I should keep the data ordered by person_id and u

# How are we going to use the cause of death? do not predict cuase of death. I assume that most causes are in the diagnosis.
# Would they work well as auxiliary labels?
# I cannot make those judgments. I will use deaths as auxiliary labels.
# The performance of the model should not be judged by the auxiliary label performance, unless we can judge that the diagnosis files don't contain cause of deaths. Otherwise it's useless.

# I cannot even enforce that the person and timestamp would be unique, because one day can have hospitalization, diagnosis and prescription, multiple records.
# All time series are finitely sampled. Just feed them one by one. The problem is that I cannot make sure that the time interval that I setup would be good. More refined, sparse signals. More ambiguous, then multiple records on the same day.
# This really does not make much sense. We cannot combine records into monthly based? We can. It all depends on how sensitive our machine is.


# okay, let's combine the time series to have a preicision of 1 month.

# in the cause of death file, are there patients that use different code types for two codes?
hello<-main %>% filter(!is.na(code_type)) %>% group_by(rep_person_id, code_type) %>% select(rep_person_id,code_type)
hello<-unique(hello)%>% group_by(rep_person_id) %>% mutate(n=n())
# 2% patients have multiple codings.
table(hello$n)
# I have no time to deal with HIC, I will remove them all
> table(hello$code_type)

  BRK   HIC ICD10  ICD9
    1   229 18533    29

# this is for string to date conversion, I made sure all datetime are actual dates
blah<-main$death_date %>% substr(10,24)==":00:00:00.000"
sum(!blah)

# probe for dirty rows in cause of death
# we need to try to find the dirty rows
table(cod$code_type)
# etc. the data is very clean.

# actually, we do not have the diagnoses for all the rep demos

# I looked up some lab coes in lab_src_code and lab_loinc_code, and I found that they are redundant
# most information are coded in lab_loinc_code already

# I am trying to split range, but it turns out that there is one lab that cannot be splitted elegantly. One lab only.

# this test needs to be treated specifically

# dirty split investigation
hello<-!is.na(splitted$C)
 hello<-splitted[hello,]
 hello<-setDT(hello)
hello[sample(132636,10),]
# the dirty splits are caused by negative range such as -2-2, where hypen - is ambiguous
# also there are some gender specific lab_ranges
# other than that, I don't see much from the sample.
# Given that the dirty splitts are 0.1%, it's safe to just throw them away, but cleaning it is also very simple
# This should yield a 0.01-0.001% dirty rows, within 30 minutes of work.

# well, there goes my 30 minutes
# the lab dataset is messier than you think.
# some does not have a range, but still has lab_abn_flag
# I can only deal with it case by case.
