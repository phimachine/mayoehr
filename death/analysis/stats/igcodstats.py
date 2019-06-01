# we only care about the code elimination statistics, nothing else
# two goals:
# get the post processed code frequency
# get the codes that got mapped to others

import pandas as pd
from death.post.inputgen_planJ import InputGenJ
from pathlib import Path
import pickle

pickle_path = "/infodev1/rep/projects/jason/pickle/new/"

def plotcod():
    ig = InputGenJ(elim_rare_code=True, no_underlying=True, death_only=True, debug=True)
    for dfn, col in ig.bar_separated + ig.no_bar:

        dic = ig.get_dict(dfn,col)
        countdic=ig.get_count_dict(dfn,col)
        sortdic=ig.get_sort_dic(dfn,col)
        df=pd.DataFrame({"code":list(dic.keys()), "index":list(dic.values())})
        df.to_csv("igstats/"+dfn+"_"+col+"_dict.csv")

        # the number of times the key has appeared in the final ig

        df=pd.DataFrame({"code":list(countdic.keys()), "count":list(countdic.values())})
        df.to_csv("igstats/"+dfn+"_"+col+"_count.csv")

        # sorted by the order of frequency of the keys
        df=pd.DataFrame({"code":list(sortdic.keys()), "place":list(sortdic.values())})
        df.to_csv("igstats/"+dfn+"_"+col+"_sort.csv")

    print("Done")

if __name__ == '__main__':
    plotcod()