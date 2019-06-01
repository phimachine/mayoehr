
import pandas as pd
from death.post.inputgen_planJ import InputGenJ

def plottoe():
    ig = InputGenJ(elim_rare_code=True, no_underlying=True, death_only=True, debug=True)
    death_date = ig.death["death_date"].to_frame()
    latest_visit = ig.earla["latest"].to_frame()
    death_date = death_date.reset_index()
    # ensure uniqueness
    death_date = death_date.drop_duplicates(subset="rep_person_id")
    death_date = death_date.drop(['id'], axis=1)

    merged=pd.merge(death_date, latest_visit, how='left',on=['rep_person_id'])
    # 19583, thank god
    merged=merged.dropna()
    merged["death_date"] = merged["death_date"].apply(lambda x: x.to_datetime64())
    merged["latest"] =  merged["latest"].apply(lambda x : x.to_datetime64())
    merged["toe"]= merged["death_date"]-merged["latest"]
    toe=merged["toe"].astype("timedelta64[M]").astype("int")
    toe[toe<0]=0
    toe.to_csv("toe_frequency.csv")

if __name__ == '__main__':
    plottoe()