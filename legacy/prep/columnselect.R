# This is the preprocessing file. Go to README.md for more detail.
# Jason Hu. June 2018.

require(data.table)
require(dplyr)

# Dataset selection.
# cauase of death
cod<-fread("/infodev/rep/data/cause_of_death.csv")
# the columns that I need are:
[rep_person_id, death_date, age_years, sex, res_county, hospital_patient, underlying, injury_flag, code_type, code]

# demographics 
demo<-fread("/infodev1/rep/data/demographics.dat")
# cols
# I need death_yes and death_date to see if it's a natural death. Or maybe not.
[rep_person_id,birth_year,death_yes,death_date,sex,race,ethnicity,educ_level]

# diagnosis
dia<-fread("/infodev1/rep/data/diagnosis.csv")
# columns
[rep_person_id,dx_date,dx_code_type,dx_code_seq,DX_CODE]
# what is dx_inout_code, dx_ed_code and dx_uc_code?

# hospitalization
hosp<-fread("/infodev1/rep/data/hospitalizations.dat")
# columns
[rep_person_id, hosp_admit_dt,hosp_disch_dt,hosp_inout_code,hosp_adm_source,hosp_disch_disp, hosp_primary_dx,hosp_dx_code_type,hosp_secondary_dx_1,hosp_secondary_dx_2,etc.]
# we should make a in-hospital flag given the admission and dispatch
# the information is redundant, but it's useful for the model. I want basics
# like this to be fed in early on.
# what is hosp_inout_code, hosp_ed_patient

# labs
labs<-fread("/infodev1/rep/data/labs.dat",fill=TRUE)
# columns
[rep_person_id, lab_date, lab_src_code, lab_loinc_code, lab_result, lab_range, lab_units, lab_abn_flag]
# note that lab range and lab units can be easily used to normalize the values, generally.

# prescriptions
# this will be a special file.
pres<-fread("/infodev1/rep/data/prescriptions.csv")
# columns
[rep_person_id,MED_DATE,med_generic,med_strength,med_route,med_dose,
## med_form and med_route were excluded, if included, it should be a cartesian cross with med_generic
## this file requires some processing. Frequency, duration, refills. We should place some sort of strength on the dimension that indicates the medicine.
## note that the medicines are also coded, so we need structured information too.
## you really cannot avoid NLP. one hot is not good enough.

# services
serv<-fread("/infodev1/rep/data/services.dat")
# columns
[rep_person_id, SRV_DATE, srv_month, srv_px_code_type, srv_px_count, srv_px_code, SRV_LOCATION, srv_quantity, srv_age_years, SRV_ADMIT_DATE, srv_admit_type, srv_admit_src, SRV_DISCH_DATE, srv_disch_stat]
# what is srv_px_mod
# srv_age_years can be redundant. Can we test whether the model thinks it's redundant?
# we need structured information about srv_px_code. This is different. We have srv_px_code_desc, again, we need to mine it.

# surgeries
surg<-fread("/infodev1/rep/data/surgeries.dat")
# columns
[rep_person_id,px_date,px_codetype,px_code]
# I'm starting to think that location might matter. We probably don't need zipcode, but px_loc_code would help? You cannot rule it out.

# tobacco_status
toba<-fread("/infodev1/rep/data/tobacco_status.dat")
# this is a file that requires processing similar to prescription
# the information here can be helpful to some diseases
# the advice we provide about smoking would be helpful too
# but at the moment this is in a survey format, we pretty much cannot use it at all.

# vitals
vitals<-fread("/infodev1/rep/data/vitals.dat")
# columns
[rep_person_id,VITAL_DATE,vital_name,vital_value_num,VITAL_UNIT,vital_seq,vital_src_code]
# what is vital_seq?
