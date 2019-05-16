# make latex table
# no, this does not work.
from death.post.inputgen_planJ import InputGenJ
ig=InputGenJ(cached=False)

for dfn in ig.dfn:
    df=getattr(ig, dfn)
    print(dfn)
    print(df.to_latex())