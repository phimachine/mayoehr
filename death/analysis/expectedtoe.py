from death.post.inputgen_planJ import InputGenJ
import pandas as pd
import scipy
import scipy.stats
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from scipy.optimize import minimize

# https://stackoverflow.com/questions/30299267/geometric-median-of-multidimensional-points
def onedgeomedian(array):
    x0 = np.array([sum(array) / len(array)])
    def dist_func(x0):
        return sum(((np.full(len(array), x0[0]) - array) ** 2) ** (1 / 2))

    res = minimize(dist_func, x0, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})
    print(res)

def calcexptoe():
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
    # 7.233
    print(toe.mean())
    # smoothl1loss
    #11.276917505612655
    print((toe - toe.mean()).abs().apply(lambda x: 0.5 * x * x if x < 1 else x - 0.5).mean())
    print("we done")

    onedgeomedian(toe)
    # the geo median is 0
    # 7.423658275034469
    print(toe.abs().apply(lambda x: 0.5 * x * x if x < 1 else x - 0.5).mean())

    toe[toe<0]=0
    onedgeomedian(toe)

    # still zero. well.
    print("DONE")

def test_onedgeomedian():
    np.random.seed(3)
    test_array = np.array([np.random.normal(88.2, 20) for i in np.arange(10000)])
    onedgeomedian(test_array)

