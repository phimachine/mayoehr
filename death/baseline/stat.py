# this file contains all the statistical models for baselines.
# I chose to do all of these in Python, not R, because the data is finalized in Python, not R. We can import from
# python to R, but it seems unreasonable.

from death.post.inputgen_planD import InputGenD, train_valid_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Logistic regression.
# The goal is bunch of binary values.

ig = InputGenD()
trainds, validds = train_valid_split(ig, split_fold=10)


lr=LogisticRegression(random_state=0,multi_class="ovr")

# X is a nd array with (150,4)
# y is a nd array with (150)
X, y = load_iris(return_X_y=True)
# lrfit=lr.fit(X,y)



print("end")