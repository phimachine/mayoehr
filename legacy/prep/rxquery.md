# notes
getApproximateMatch(drug, 1, 0)
findRxcuitByString(name)
getRelatedByType(cui, ["IN"])


# the plan of the query is to query with the med_name, this query will cover 95% of the missing entries
# for the last 230,000 entries we will do it by descriptor and approximate string match
# this will be computationally intensive for them. kudos to nih for running this service free.
