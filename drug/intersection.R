# Title     : TODO
# Objective : TODO
# Created by: JasonHu
# Created on: 2/2/2019

require('data.table')
require('dplyr')
drug_deaths<-fread("/infodev1/rep/projects/jason/drugrepid.csv")
drug_users<- fread('/infodev1/rep/projects/jason/drug_users.txt')

intersect(drug_users, drug_deaths)
# produces 177 results
# now, if the two dfs are independent, it would yield 74 results by expectation. this is nonsense.
# 230,000 total, 14032 users, 1220 deaths, 177 intersections.
# I back examined patient 1, I see drug usages and drug related death. I don't know what to say. It looks right.